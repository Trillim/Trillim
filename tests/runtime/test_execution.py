from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from pydantic import JsonValue

from trillim.compiler.artifacts.artifact_pydantic_models import (
    _ExecutableManifest,
    _WorkloadPlan,
)
from trillim.compiler.dtype import _QUANTIZED_DTYPES, _decode, _encode
from trillim.runtime import ExecutionError, Runtime
from trillim.tdl.core.types import DType


def test_workload_guards_and_strided_output_ownership(
    compile_tdl: Callable[[str], Path],
) -> None:
    directory = compile_tdl("""
        from trillim.tdl import Graph, Input, Output, Permute

        with Graph() as graph:
            x = Input(name="x", shape=(None, 3), maximum_shape=(4, 3), dtype="fp32")
            Output(Permute(x, (1, 0)), name="result")
    """)
    path = directory / "executable.json"
    manifest = _ExecutableManifest.model_validate_json(path.read_bytes())
    size: JsonValue = {"op": "dimension", "tensor": manifest.inputs["x"], "axis": 0}
    guard: dict[str, JsonValue] = {
        "op": "guard",
        "left": size,
        "right": 3,
        "predicate": "eq",
        "reason": "three-row workload plan",
    }
    manifest.plans.append(
        _WorkloadPlan(
            minimum_tokens=3,
            maximum_tokens=3,
            arena_size=manifest.arena_size,
            buffers=manifest.buffers,
            program=[guard, *manifest.program],
            expressions=manifest.expressions,
        )
    )
    manifest.program.insert(0, {**guard, "right": 2, "reason": "two-row fallback plan"})
    path.write_text(manifest.model_dump_json())

    source = np.arange(18, dtype=np.float32).reshape(3, 6)[:, ::2]
    expected = source.T.copy()
    with Runtime(directory) as runtime:
        with pytest.raises(ValueError):
            runtime.run({"x": np.zeros((5, 3), dtype=np.float32)})
        with pytest.raises(ValueError):
            runtime.run({"x": np.zeros((2, 2), dtype=np.float32)})
        first = runtime.run({"x": source})["result"]
        source.fill(-1)
        second = runtime.run({"x": source[:2]})["result"]
        np.testing.assert_array_equal(second, source[:2].T)
        second.fill(99)
    assert first.flags.owndata
    np.testing.assert_array_equal(first, expected)


def test_loop_swaps_state_and_preserves_reduction_identities(
    compile_tdl: Callable[[str], Path],
) -> None:
    directory = compile_tdl("""
        from trillim.tdl import Add, Compare, Constant, Graph, If, Input, Mul, Output, Reduce, While

        with Graph() as graph:
            x = Input(name="x", shape=(2, None), maximum_shape=(2, 4), dtype="fp32")
            count = Input(name="count", shape=(), dtype="int64", minimum=0, maximum=3)
            negate = Input(name="negate", shape=(), dtype="bool")
            values = If(negate, then_branch=lambda: (Mul(x, -1.0),), else_branch=lambda: (x,))
            total, energy = Reduce(
                (values[0], values[0]),
                initial=(Constant(10.0), Constant(20.0)),
                axes=(1,),
                reducer=lambda a, b: (Add(a[0], b[0]), Add(a[1], Mul(b[1], b[1]))),
            )
            _, left, right = While(
                (Constant(0), total, energy),
                condition=lambda state: Compare(state[0], count, predicate="lt"),
                body=lambda state: (Add(state[0], 1), state[2], state[1]),
            )
            Output(left, name="left")
            Output(right, name="right")
    """)
    x = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.float32)
    with Runtime(directory) as runtime:
        even = runtime.run({"x": x, "count": 2, "negate": False})
        np.testing.assert_array_equal(even["left"], [12, 13])
        np.testing.assert_array_equal(even["right"], [34, 97])
        odd = runtime.run({"x": x, "count": 3, "negate": True})
        np.testing.assert_array_equal(odd["left"], [34, 97])
        np.testing.assert_array_equal(odd["right"], [8, 7])
        empty = runtime.run({"x": x[:, :0], "count": 0, "negate": True})
        np.testing.assert_array_equal(empty["left"], [10, 10])
        np.testing.assert_array_equal(empty["right"], [20, 20])


def test_cache_snapshots_and_sliding_eviction(
    compile_tdl: Callable[[str], Path],
) -> None:
    directory = compile_tdl("""
        from trillim.tdl import Cache, Graph, Input, Output

        with Graph() as graph:
            x = Input(name="x", shape=(2, None), maximum_shape=(2, 3), dtype="int32")
            sliding = Cache(name="sliding", shape=(2, 3), dtype="int32", mode="sliding", axis=1)
            Output(sliding.read(), name="before")
            sliding.append(x)
            Output(sliding.read(), name="after")
    """)
    first = np.array([[1, 2], [11, 12]], dtype=np.int32)
    second = np.array([[3, 4], [13, 14]], dtype=np.int32)
    full = np.arange(6, dtype=np.int32).reshape(2, 3)
    with Runtime(directory) as runtime:
        outputs = runtime.run({"x": first})
        assert outputs["before"].shape == (2, 0)
        np.testing.assert_array_equal(outputs["after"], first)
        outputs = runtime.run({"x": second})
        np.testing.assert_array_equal(outputs["before"], first)
        retained = np.concatenate((first, second), axis=1)[:, -3:]
        np.testing.assert_array_equal(outputs["after"], retained)
        outputs = runtime.run({"x": full})
        np.testing.assert_array_equal(outputs["before"], retained)
        np.testing.assert_array_equal(outputs["after"], full)


@pytest.mark.parametrize("dtype", sorted(_QUANTIZED_DTYPES))
def test_packed_cache_preserves_quantized_rows(
    compile_tdl: Callable[[str], Path], dtype: DType
) -> None:
    directory = compile_tdl(f"""
        from trillim.tdl import Cache, Graph, Input, Output

        with Graph() as graph:
            x = Input(name="x", shape=(2, 35), dtype="fp32")
            cache = Cache(name="cache", shape=(2, 35), dtype={dtype!r})
            Output(cache.read(), name="before")
            cache.assign(x)
            Output(cache.read(), name="after")
    """)
    x = (np.arange(70, dtype=np.float32).reshape(2, 35) % 19 - 9) / 8
    y = x * np.float32(-0.5)
    expected = _decode(_encode(x, dtype), x.shape)
    with Runtime(directory) as runtime:
        first = runtime.run({"x": x})
        second = runtime.run({"x": y})
    np.testing.assert_array_equal(first["before"], np.zeros_like(x))
    np.testing.assert_array_equal(first["after"], expected)
    np.testing.assert_array_equal(second["before"], expected)
    np.testing.assert_array_equal(second["after"], _decode(_encode(y, dtype), y.shape))


def test_cache_initial_values_preserve_integer_and_half_precision(
    compile_tdl: Callable[[str], Path],
) -> None:
    directory = compile_tdl("""
        from trillim.tdl import Cache, Graph, Output

        with Graph() as graph:
            integer = Cache(name="integer", shape=(2,), dtype="uint64", initial=2**64 - 3)
            half = Cache(name="half", shape=(2,), dtype="fp16", initial=1 + 2**-11 + 2**-35)
            Output(integer.read(), name="integer")
            Output(half.read(), name="half")
    """)
    with Runtime(directory) as runtime:
        actual = runtime.run({})
    uint64 = np.full(2, 2**64 - 3, dtype=np.uint64)
    fp16 = np.full(2, np.nextafter(np.float16(1), np.float16(2)), dtype=np.float16)
    np.testing.assert_array_equal(actual["integer"], uint64)
    np.testing.assert_array_equal(actual["half"], fp16)


@pytest.mark.parametrize("dtype", ["dubgrp_int6", "grp_ternary"])
def test_packed_output_uses_canonical_decoder(
    compile_tdl: Callable[[str], Path], dtype: DType
) -> None:
    directory = compile_tdl(f"""
        import numpy as np
        from trillim.tdl import Constant, Graph, Output

        with Graph() as graph:
            Output(Constant(np.zeros(1, dtype=np.float32), dtype={dtype!r}), name="out")
    """)
    manifest = _ExecutableManifest.model_validate_json(
        (directory / "executable.json").read_bytes()
    )
    record = manifest.buffers[manifest.outputs["out"]]
    assert record.storage == "constant"
    with (directory / manifest.constants).open("r+b") as constants:
        constants.seek(record.offset)
        if dtype == "dubgrp_int6":
            # Code 31 represents -1. Both scales are positive zero.
            low = b"\x0f" + bytes(127)
            high = b"\x01" + bytes(63)
            constants.write(low + bytes(2) + high + bytes(16))
        else:
            # The high two bits of the first byte encode the first ternary value.
            first_byte = constants.read(1)
            constants.seek(record.offset)
            constants.write(bytes([first_byte[0] | 0xC0]))

    with Runtime(directory) as runtime:
        if dtype == "dubgrp_int6":
            output = runtime.run({})["out"]
            assert output.dtype == np.float32
            assert output.view(np.uint32).item() == 0x80000000
        else:
            with pytest.raises(ExecutionError, match="ternary code 3"):
                runtime.run({})


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_integer_shape_guard_preserves_values_above_float_precision(
    compile_tdl: Callable[[str], Path], dtype: str
) -> None:
    exact = 2**53 + 1 if dtype == "int64" else 2**64 - 2
    directory = compile_tdl(f"""
        from trillim.tdl import Graph, Input, Output, Permute

        with Graph() as graph:
            count = Input(name="count", shape=(), dtype={dtype!r}, minimum={exact}, maximum={exact + 1})
            x = Input(name="x", shape=(2, 2), dtype={dtype!r})
            Output(Permute(x, (1, 0)), name="result")
            Output(count, name="count")
    """)
    limits = np.iinfo(dtype)
    values = np.array([[limits.min, exact], [exact - 1, limits.max]], dtype=dtype)
    with Runtime(directory) as runtime:
        actual = runtime.run({"x": values, "count": exact})
        assert actual["count"].item() == exact
        assert actual["result"].dtype == values.dtype
        np.testing.assert_array_equal(actual["result"], values.T)
        if dtype == "uint64":
            mixed = runtime.run({"x": [[0, 2**63 + 1], [2**64 - 1, 1]], "count": exact})
            expected = np.array([[0, 2**64 - 1], [2**63 + 1, 1]], dtype=np.uint64)
            assert mixed["result"].dtype == expected.dtype
            np.testing.assert_array_equal(mixed["result"], expected)
        with pytest.raises(ExecutionError, match="count: input value"):
            runtime.run({"x": values, "count": exact - 1})
