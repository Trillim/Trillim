from __future__ import annotations

import json
import math
import re
import struct

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from pydantic import JsonValue, TypeAdapter

from tests.backends.cpu.harness import NativeCPU, Response, result
from tests.backends.cpu.strategies import INTEGERS, JSON_VALUES, TEXT, assert_json_equal

_VALUE: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def reject_constant(text: str) -> None:
    raise ValueError(text)


def finite_float(text: str) -> float:
    value = float(text)
    significand = text.lower().partition("e")[0]
    if (
        not math.isfinite(value)
        or value == 0
        and any(c in "123456789" for c in significand)
    ):
        raise ValueError("float outside representable range")
    return value


def unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    fields: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("duplicate key")
        fields[key] = value
    return fields


def validate_tree(value: JsonValue, depth: int = 0) -> None:
    if depth > 256:
        raise ValueError("nesting limit exceeded")
    if isinstance(value, str):
        value.encode("utf-8")
    elif isinstance(value, list):
        for item in value:
            validate_tree(item, depth + 1)
    elif isinstance(value, dict):
        for key, item in value.items():
            key.encode("utf-8")
            validate_tree(item, depth + 1)


def reference(payload: bytes) -> JsonValue:
    value = _VALUE.validate_python(
        json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=unique_object,
        )
    )
    validate_tree(value)
    return value


def assert_rejected(response: Response, payload: bytes) -> None:
    assert not response.ok, payload
    assert response.kind == "invalid"
    match = re.search(r"at byte (\d+):", response.error)
    assert match is not None, response.error
    assert 0 <= int(match[1]) <= len(payload)


def compare_reference(native: NativeCPU, payload: bytes) -> None:
    response = native.request(b"j", payload)
    try:
        expected = reference(payload)
    except (ValueError, UnicodeError):
        assert_rejected(response, payload)
    else:
        assert_json_equal(expected, result(response))


@settings(max_examples=1000, deadline=None)
@example(value={"\x00": [-(2**127), 2**64, "中文 😀", -0.0]}, ascii=False, mode=b"v")
@example(value=[5e-324, -5e-324, 1.7976931348623157e308], ascii=True, mode=b"j")
@given(value=JSON_VALUES, ascii=st.booleans(), mode=st.sampled_from((b"j", b"v")))
def test_json_roundtrip_preserves_types_and_float_bits(
    native_cpu: NativeCPU, value: JsonValue, ascii: bool, mode: bytes
) -> None:
    payload = json.dumps(value, ensure_ascii=ascii, allow_nan=False).encode("utf-8")
    response = native_cpu.request(mode, payload)
    assert_json_equal(value, result(response))
    again = native_cpu.request(mode, response.result.encode("utf-8"))
    assert_json_equal(value, result(again))


@settings(max_examples=300, deadline=None)
@example(text='\x00\b\f\n\r\t"\\/\U0010ffff')
@given(text=TEXT)
def test_unicode_and_escaping(native_cpu: NativeCPU, text: str) -> None:
    for ascii in (False, True):
        payload = json.dumps(text, ensure_ascii=ascii).encode("utf-8")
        assert result(native_cpu.request(b"j", payload)) == text
    assert result(native_cpu.request(b"s", text.encode("utf-8"))) == text


@settings(max_examples=300, deadline=None)
@given(
    value=INTEGERS,
    before=st.text(" \t\r\n", max_size=8),
    after=st.text(" \t\r\n", max_size=8),
)
def test_arbitrary_integer_tokens_and_whitespace(
    native_cpu: NativeCPU, value: int, before: str, after: str
) -> None:
    actual = result(native_cpu.request(b"j", f"{before}{value}{after}".encode()))
    assert type(actual) is int and actual == value


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        b" ",
        b"\0",
        b"1\0",
        b"[",
        b"{",
        b"[1,]",
        b'{"x":1,}',
        b"NaN",
        b"Infinity",
        b"-Infinity",
        b"01",
        b"-01",
        b"+1",
        b".1",
        b"1.",
        b"1e",
        b"1e+",
        b"1e-",
        b"1e309",
        b"1e-10000",
        b"1 2",
        b"true false",
        b"True",
        b"nul",
        b"/*comment*/null",
        b"{x:1}",
        b'"\\x41"',
        b'"\\u123"',
        b'"\\uZZZZ"',
        b'"unterminated',
        b'"abc\\',
        b'"\\v"',
    ),
)
def test_malformed_json_is_rejected(native_cpu: NativeCPU, payload: bytes) -> None:
    assert_rejected(native_cpu.request(b"j", payload), payload)


@settings(max_examples=200, deadline=None)
@given(key=TEXT, left=JSON_VALUES, right=JSON_VALUES)
def test_duplicate_keys_are_rejected_after_unescaping(
    native_cpu: NativeCPU, key: str, left: JsonValue, right: JsonValue
) -> None:
    utf16 = key.encode("utf-16-be")
    escaped = (
        '"'
        + "".join(
            f"\\u{int.from_bytes(utf16[i : i + 2], 'big'):04x}"
            for i in range(0, len(utf16), 2)
        )
        + '"'
    )
    payload = f"{{{json.dumps(key)}:{json.dumps(left)},{escaped}:{json.dumps(right)}}}".encode()
    response = native_cpu.request(b"j", payload)
    assert_rejected(response, payload)
    assert "duplicate" in response.error


@pytest.mark.parametrize(
    "raw",
    (
        b"\xc2",
        b"\xe2\x82",
        b"\xf0\x90\x80",
        b"\xc0\x80",
        b"\xe0\x80\x80",
        b"\xed\xa0\x80",
        b"\xf0\x80\x80\x80",
        b"\xf4\x90\x80\x80",
        b"\xe2A\x80",
    ),
)
def test_invalid_utf8_sequences_are_rejected(native_cpu: NativeCPU, raw: bytes) -> None:
    payload = b'"' + raw + b'"'
    assert_rejected(native_cpu.request(b"j", payload), payload)
    response = native_cpu.request(b"s", raw)
    assert not response.ok and response.kind == "invalid"


@settings(max_examples=200, deadline=None)
@given(
    lead=st.sampled_from((*range(0x80, 0xC2), *range(0xF5, 0x100))),
    tail=st.binary(max_size=20),
)
def test_illegal_utf8_leading_bytes(
    native_cpu: NativeCPU, lead: int, tail: bytes
) -> None:
    payload = b'"' + bytes((lead,)) + tail + b'"'
    assert_rejected(native_cpu.request(b"j", payload), payload)


@given(code=st.integers(0, 31))
def test_raw_control_characters_require_escaping(
    native_cpu: NativeCPU, code: int
) -> None:
    payload = b'"before' + bytes((code,)) + b'after"'
    assert_rejected(native_cpu.request(b"j", payload), payload)


@settings(max_examples=200, deadline=None)
@given(high=st.integers(0xD800, 0xDBFF), low=st.integers(0xDC00, 0xDFFF))
def test_surrogate_pairs_and_orphans(
    native_cpu: NativeCPU, high: int, low: int
) -> None:
    payload = f'"\\u{high:04x}\\u{low:04x}"'.encode()
    expected = chr(0x10000 + ((high - 0xD800) << 10) + low - 0xDC00)
    assert result(native_cpu.request(b"j", payload)) == expected
    for text in (f'"\\u{high:04x}"', f'"\\u{low:04x}"', f'"\\u{high:04x}\\u0041"'):
        payload = text.encode()
        assert_rejected(native_cpu.request(b"j", payload), payload)


@pytest.mark.parametrize("depth", (0, 1, 255, 256, 257, 300))
@pytest.mark.parametrize("object", (False, True))
def test_nesting_limit(native_cpu: NativeCPU, depth: int, object: bool) -> None:
    before, after = (b'{"x":', b"}") if object else (b"[", b"]")
    payload = before * depth + b"0" + after * depth
    response = native_cpu.request(b"j", payload)
    if depth > 256:
        assert_rejected(response, payload)
    else:
        assert response.ok, response.error
        assert response.result.encode() == payload


@pytest.mark.parametrize("depth", (0, 1, 255, 256, 257, 300))
def test_serializer_checks_nesting_limit(native_cpu: NativeCPU, depth: int) -> None:
    response = native_cpu.call(b"o", {"op": "nest", "depth": depth})
    if depth > 256:
        assert not response.ok and response.kind == "invalid"
        assert "nesting limit" in response.error
    else:
        assert response.ok, response.error
        assert response.result == "[" * depth + "0" + "]" * depth


@st.composite
def mutated_document(draw: st.DrawFn) -> bytes:
    value = draw(JSON_VALUES)
    ascii = draw(st.booleans())
    payload = bytearray(json.dumps(value, ensure_ascii=ascii, allow_nan=False).encode())
    index = draw(st.integers(0, len(payload) - 1))
    operation = draw(st.sampled_from(("delete", "insert", "replace", "truncate")))
    byte = draw(st.integers(0, 255))
    if operation == "delete":
        del payload[index]
    elif operation == "insert":
        payload.insert(index, byte)
    elif operation == "replace":
        payload[index] = byte
    else:
        del payload[index:]
    return bytes(payload)


@settings(max_examples=1000, deadline=None)
@given(payload=mutated_document())
def test_mutations_match_strict_reference(
    native_cpu: NativeCPU, payload: bytes
) -> None:
    compare_reference(native_cpu, payload)


@settings(max_examples=500, deadline=None)
@given(payload=st.binary(max_size=512))
def test_arbitrary_bytes_match_strict_reference(
    native_cpu: NativeCPU, payload: bytes
) -> None:
    compare_reference(native_cpu, payload)


@settings(max_examples=300, deadline=None)
@given(
    object=st.one_of(st.none(), st.dictionaries(TEXT, JSON_VALUES, max_size=5)),
    array=st.lists(JSON_VALUES, max_size=5),
    key=TEXT,
    value=JSON_VALUES,
)
def test_object_and_array_operations(
    native_cpu: NativeCPU,
    object: dict[str, JsonValue] | None,
    array: list[JsonValue],
    key: str,
    value: JsonValue,
) -> None:
    fields = {} if object is None else object.copy()
    found = key in fields
    before = fields.get(key)
    fallback = fields.get(key, "missing")
    fields[key] = value
    expected: dict[str, JsonValue] = {
        "found": found,
        "contains": found,
        "before": before,
        "fallback": fallback,
        "object": fields,
        "original": object,
        "rebuilt_equal": True,
        "array": [*array, value],
        "reverse": [value, *reversed(array)],
        "last": value,
        "object_size": len(fields),
    }
    actual = result(
        native_cpu.call(
            b"o",
            {
                "op": "mutate",
                "object": object,
                "array": array,
                "key": key,
                "value": value,
            },
        )
    )
    assert_json_equal(expected, actual)


@pytest.mark.parametrize(
    "width,signed",
    ((8, True), (8, False), (32, True), (32, False), (64, True), (64, False)),
)
@settings(max_examples=200, deadline=None)
@given(value=INTEGERS)
def test_integer_get_checks_destination_range(
    native_cpu: NativeCPU, width: int, signed: bool, value: int
) -> None:
    target = f"{'int' if signed else 'uint'}{width}"
    response = native_cpu.call(b"o", {"op": target, "value": value})
    low = -(2 ** (width - 1)) if signed else 0
    high = 2 ** (width - int(signed)) - 1
    if low <= value <= high:
        actual = result(response)
        assert type(actual) is int and actual == value
    else:
        assert not response.ok and response.kind == "range"


@pytest.mark.parametrize("target", ("int64", "uint64", "exact_integer"))
@settings(max_examples=100, deadline=None)
@given(
    value=st.one_of(
        st.none(),
        TEXT,
        st.floats(allow_nan=False, allow_infinity=False),
        st.lists(st.integers(), max_size=3),
    )
)
def test_integer_get_rejects_non_integer_values(
    native_cpu: NativeCPU, target: str, value: JsonValue
) -> None:
    response = native_cpu.call(b"o", {"op": target, "value": value})
    assert not response.ok


@settings(max_examples=300, deadline=None)
@given(value=st.one_of(INTEGERS, st.floats(allow_nan=False, allow_infinity=False)))
def test_json_double_conversion_matches_python(
    native_cpu: NativeCPU, value: float
) -> None:
    response = native_cpu.call(b"o", {"op": "double", "value": value})
    try:
        expected = float(value)
    except OverflowError:
        assert not response.ok and response.kind == "range"
    else:
        assert_json_equal(expected, result(response))


@settings(max_examples=300, deadline=None)
@example(bits=0x7FF0_0000_0000_0000)
@example(bits=0xFFF0_0000_0000_0000)
@example(bits=0x7FF8_0000_0000_0000)
@example(bits=0x8000_0000_0000_0000)
@example(bits=1)
@given(bits=st.integers(0, 2**64 - 1))
def test_constructed_doubles_serialize_finite_or_null(
    native_cpu: NativeCPU, bits: int
) -> None:
    value = struct.unpack("<d", struct.pack("<Q", bits))[0]
    actual = result(native_cpu.call(b"o", {"op": "real_bits", "bits": bits}))
    assert_json_equal(value if math.isfinite(value) else None, actual)
