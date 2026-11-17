from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from pydantic import JsonValue

from tests.compiler.compiler_dataclasses import PlannedCase
from tests.compiler.model_cases import CASE_NAMES
from tests.compiler.planning import emit
from trillim.compiler.artifacts.artifact_pydantic_models import _ExecutableManifest
from trillim.compiler.passes.verify import _verify
from trillim.runtime import Runtime


@pytest.mark.parametrize("name", CASE_NAMES)
def test_model_artifact_round_trip(
    name: str, model_plans: Callable[[str], PlannedCase]
) -> None:
    planned = model_plans(name)
    emitted = emit(planned)
    verified = _verify(planned.directory)
    assert emitted == verified
    assert verified.version == verified.abi_version == 1
    assert verified.buffers[verified.outputs["result"]].maximum_shape == list(
        planned.case.output_shape
    )
    with Runtime(planned.directory, threads=1) as runtime:
        result = runtime.run(planned.case.inputs)["result"]
    assert result.shape == planned.case.output_shape
    assert np.isfinite(result).all()
    if planned.case.oracle is not None:
        np.testing.assert_allclose(
            result.astype(np.float32), planned.case.oracle, rtol=1e-5, atol=1e-5
        )


def _first_call(manifest: _ExecutableManifest) -> dict[str, JsonValue]:
    return next(
        instruction for instruction in manifest.program if instruction["op"] == "call"
    )


@pytest.mark.parametrize(
    "damage",
    (
        "buffer_extent",
        "buffer_alignment",
        "constant_extent",
        "cyclic_alias",
        "unknown_kernel",
        "missing_parameter",
        "unknown_input",
        "binding_count",
        "input_packing",
        "scratch_extent",
        "library_escape",
        "output_unproduced",
        "unknown_expression",
        "expression_cycle",
        "unknown_instruction",
        "weight_binding",
    ),
)
def test_reject_invalid_execution_contract(
    damage: str, tmp_path: Path, model_plans: Callable[[str], PlannedCase]
) -> None:
    planned = model_plans(
        "conv_branch_reduce" if damage == "constant_extent" else "projection"
    )
    manifest = emit(planned)
    shutil.copytree(planned.directory, tmp_path, dirs_exist_ok=True)
    call = _first_call(manifest)
    output = manifest.outputs["result"]
    buffer = manifest.buffers[output]
    if damage == "buffer_extent":
        manifest.buffers[output] = buffer.model_copy(
            update={"size": manifest.arena_size + 64}
        )
    elif damage == "buffer_alignment":
        manifest.buffers[output] = buffer.model_copy(update={"alignment": 3})
    elif damage == "constant_extent":
        constant = next(
            n for n, b in manifest.buffers.items() if b.storage == "constant"
        )
        manifest.buffers[constant] = manifest.buffers[constant].model_copy(
            update={"size": 10**8}
        )
    elif damage == "cyclic_alias":
        manifest.buffers[output] = buffer.model_copy(
            update={"storage": "alias", "source": output}
        )
    elif damage == "unknown_kernel":
        call["kernel"] = "missing"
    elif damage == "missing_parameter":
        call["parameters"] = {}
    elif damage == "unknown_input":
        call["inputs"] = ["missing", "missing"]
    elif damage == "binding_count":
        call["outputs"] = [output, output]
    elif damage == "input_packing":
        call["input_packings"] = ["row_int8_block32", "canonical"]
    elif damage == "scratch_extent":
        call["scratch"] = {"offset": manifest.arena_size, "size": 64}
    elif damage == "library_escape":
        kernel = next(iter(manifest.kernels.values()))
        kernel["library"] = "../outside.so"
    elif damage == "output_unproduced":
        manifest = manifest.model_copy(update={"program": []})
    elif damage == "unknown_expression":
        call["parameters"] = {"m": {"expr": "missing"}}
    elif damage == "expression_cycle":
        manifest.expressions["cycle"] = {"expr": "cycle"}
    elif damage == "weight_binding":
        name = next(n for n, b in manifest.buffers.items() if b.storage == "weight")
        manifest.buffers[name] = manifest.buffers[name].model_copy(
            update={"dtype": "fp16"}
        )
    elif damage == "unknown_instruction":
        call["op"] = "invalid"
    else:
        raise AssertionError(damage)
    (tmp_path / "executable.json").write_text(manifest.model_dump_json())
    with pytest.raises((ValueError, TypeError)):
        _verify(tmp_path)


@pytest.mark.parametrize(
    "damage",
    (
        "offset",
        "overlap",
        "negative_capacity",
        "condition_type",
        "loop_bound",
        "if_arity",
        "reduce_arity",
        "cache_snapshot",
        "cache_action",
    ),
)
def test_control_and_cache_contract(
    damage: str, tmp_path: Path, model_plans: Callable[[str], PlannedCase]
) -> None:
    name = (
        "recurrence"
        if damage in {"condition_type", "loop_bound"}
        else "conv_branch_reduce"
        if damage in {"if_arity", "reduce_arity"}
        else "kv_cache"
    )
    planned = model_plans(name)
    manifest = emit(planned)
    shutil.copytree(planned.directory, tmp_path, dirs_exist_ok=True)
    if damage == "loop_bound":
        loop = next(i for i in manifest.program if i["op"] == "while")
        loop["maximum_iterations"] = -1
    elif damage == "if_arity":
        branch = next(i for i in manifest.program if i["op"] == "if")
        branch["branches"] = []
    elif damage == "reduce_arity":
        reduction = next(i for i in manifest.program if i["op"] == "reduce")
        reduction["parameters"] = []
    elif damage == "cache_snapshot":
        read = next(i for i in manifest.program if i["op"] == "cache_read")
        read["snapshot"] = "invalid"
    elif damage == "cache_action":
        write = next(i for i in manifest.program if i["op"] == "cache_write")
        write["action"] = "invalid"
    elif damage == "condition_type":
        loop = next(i for i in manifest.program if i["op"] == "while")
        condition = loop["condition"]
        assert isinstance(condition, dict)
        result = condition["result"]
        assert isinstance(result, str)
        manifest.buffers[result] = manifest.buffers[result].model_copy(
            update={"dtype": "fp32"}
        )
    else:
        key = manifest.caches["key"]
        if damage == "offset":
            key["offset"] = manifest.cache_size + 64
        elif damage == "overlap":
            manifest.caches["value"]["offset"] = key["offset"]
        elif damage == "negative_capacity":
            key["shape"] = [-1, 2, 8]
    (tmp_path / "executable.json").write_text(manifest.model_dump_json())
    with pytest.raises(ValueError):
        _verify(tmp_path)
