from __future__ import annotations

import math
import struct

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from pydantic import JsonValue

from tests.backends.cpu.harness import NativeCPU, result
from tests.backends.cpu.strategies import INTEGERS, assert_json_equal


def check_case(native: NativeCPU, left: int, right: int, bits: int) -> None:
    actual = result(
        native.call(b"i", {"left": str(left), "right": str(right), "float_bits": bits})
    )
    try:
        real: float | None = float(left)
    except OverflowError:
        real = None
    scalar = struct.unpack("<f", struct.pack("<I", bits))[0]
    expected: dict[str, JsonValue] = {
        "left": str(left),
        "right": str(right),
        "roundtrip": str(left),
        "compare": (left > right) - (left < right),
        "sum": str(left + right),
        "difference": str(left - right),
        "product": str(left * right),
        "divisions": [str(left // right), str(-(-left // right)), str(left % right)]
        if right
        else [],
        "real": real,
        "truncated": str(int(scalar)) if math.isfinite(scalar) else None,
        "int64": left if -(2**63) <= left < 2**63 else None,
        "uint64": left if 0 <= left < 2**64 else None,
        "size": left if 0 <= left < 2 ** (8 * struct.calcsize("P")) else None,
    }
    assert_json_equal(expected, actual)


@settings(max_examples=1000, deadline=None)
@example(left=-(2**127), right=-1, bits=0x7F80_0000)
@example(left=2**127 - 1, right=1, bits=0xFF80_0000)
@example(left=2**128, right=-(2**128), bits=0x7FC0_0000)
@example(left=2**4096 - 1, right=2**128 + 1, bits=0x7F7F_FFFF)
@example(left=-(2**4096), right=2**4096 - 1, bits=0xFF7F_FFFF)
@example(left=2**1024 - 2**970, right=-7, bits=1)
@example(left=2**1024 - 2**970 - 1, right=7, bits=0x8000_0001)
@example(left=0, right=0, bits=0x8000_0000)
@given(left=INTEGERS, right=INTEGERS, bits=st.integers(0, 2**32 - 1))
def test_arithmetic_and_conversions_match_python(
    native_cpu: NativeCPU, left: int, right: int, bits: int
) -> None:
    check_case(native_cpu, left, right, bits)


@settings(max_examples=300, deadline=None)
@given(
    mantissa=st.integers(2**52, 2**53 - 1),
    shift=st.integers(76, 970),
    offset=st.sampled_from((-1, 0, 1)),
    sign=st.sampled_from((-1, 1)),
)
def test_big_integer_to_double_rounds_ties_to_even(
    native_cpu: NativeCPU, mantissa: int, shift: int, offset: int, sign: int
) -> None:
    value = sign * ((mantissa << shift) + (1 << (shift - 1)) + offset)
    check_case(native_cpu, value, 1, 0)


@settings(max_examples=300, deadline=None)
@example(
    initial=2**127 - 1,
    operations=[("add", 1), ("sub", 2**127), ("mul", 2**128), ("add", -1)],
)
@example(initial=-(2**127), operations=[("floor", -1), ("sub", 2**127), ("mul", 0)])
@given(
    initial=st.integers(-(2**512), 2**512),
    operations=st.lists(
        st.tuples(
            st.sampled_from(("add", "sub", "mul", "floor", "ceil", "mod")),
            st.integers(-(2**256), 2**256),
        ),
        min_size=1,
        max_size=16,
    ),
)
def test_operation_sequences_cross_small_and_large_representations(
    native_cpu: NativeCPU, initial: int, operations: list[tuple[str, int]]
) -> None:
    value = initial
    steps: list[JsonValue] = []
    history: list[JsonValue] = []
    for name, operand in operations:
        if name in {"floor", "ceil", "mod"} and operand == 0:
            operand = 1
        steps.append({"op": name, "value": str(operand)})
        match name:
            case "add":
                value += operand
            case "sub":
                value -= operand
            case "mul":
                value *= operand
            case "floor":
                value //= operand
            case "ceil":
                value = -(-value // operand)
            case "mod":
                value %= operand
            case _:
                raise AssertionError(name)
        history.append(str(value))
    actual = result(native_cpu.call(b"i", {"left": str(initial), "operations": steps}))
    assert_json_equal(history, actual)


@settings(max_examples=300, deadline=None)
@given(value=INTEGERS, zeros=st.integers(0, 60))
def test_decimal_parsing_normalizes_sign_and_leading_zeros(
    native_cpu: NativeCPU, value: int, zeros: int
) -> None:
    text = ("-" if value < 0 else "") + "0" * zeros + str(abs(value))
    assert result(native_cpu.request(b"p", text.encode())) == str(value)


@settings(max_examples=150, deadline=None)
@given(
    value=INTEGERS, bad=st.sampled_from(("+", "_", "x", "\0", ".", "e", " ", "é", "١"))
)
def test_non_decimal_integer_literals_are_rejected(
    native_cpu: NativeCPU, value: int, bad: str
) -> None:
    response = native_cpu.request(b"p", (str(value) + bad).encode())
    assert not response.ok and response.kind == "invalid"


@pytest.mark.parametrize(
    "text", ("", "-", "+1", "--1", "1-2", "1.0", "1e2", " 1", "1 ", "-0x1")
)
def test_malformed_integer_literals(native_cpu: NativeCPU, text: str) -> None:
    response = native_cpu.request(b"p", text.encode())
    assert not response.ok and response.kind == "invalid"


@pytest.mark.parametrize("mode", ("floor", "ceil", "mod"))
@settings(max_examples=100, deadline=None)
@given(value=INTEGERS)
def test_division_by_zero_is_rejected(
    native_cpu: NativeCPU, mode: str, value: int
) -> None:
    response = native_cpu.call(b"i", {"op": mode, "left": str(value), "right": "0"})
    assert not response.ok and response.kind == "invalid"
    assert "division by zero" in response.error


@settings(max_examples=300, deadline=None)
@given(value=st.one_of(INTEGERS, st.booleans()))
def test_manifest_integer_conversion_is_exact(
    native_cpu: NativeCPU, value: int | bool
) -> None:
    assert result(
        native_cpu.call(b"o", {"op": "exact_integer", "value": value})
    ) == str(int(value))
