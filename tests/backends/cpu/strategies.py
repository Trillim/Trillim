from __future__ import annotations

import struct

from hypothesis import strategies as st
from pydantic import JsonValue

TEXT = st.text(st.characters(exclude_categories=["Cs"]), max_size=40)


@st.composite
def boundary_integer(draw: st.DrawFn) -> int:
    bit = draw(
        st.sampled_from(
            (
                7,
                8,
                15,
                16,
                31,
                32,
                63,
                64,
                127,
                128,
                129,
                255,
                256,
                511,
                1023,
                1024,
                4096,
            )
        )
    )
    offset = draw(st.integers(-2, 2))
    sign = draw(st.sampled_from((-1, 1)))
    return sign * ((1 << bit) + offset)


INTEGERS = st.one_of(
    st.integers(-100, 100),
    st.integers(-(2**128), 2**128),
    boundary_integer(),
    st.integers(-(2**4096), 2**4096),
)


def containers(children: st.SearchStrategy[JsonValue]) -> st.SearchStrategy[JsonValue]:
    return st.one_of(
        st.lists(children, max_size=6), st.dictionaries(TEXT, children, max_size=6)
    )


JSON_VALUES: st.SearchStrategy[JsonValue] = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        INTEGERS,
        st.floats(allow_nan=False, allow_infinity=False),
        TEXT,
    ),
    containers,
    max_leaves=30,
)


def assert_json_equal(expected: JsonValue, actual: JsonValue) -> None:
    assert type(actual) is type(expected), (expected, actual)
    if isinstance(expected, float):
        assert isinstance(actual, float)
        assert struct.pack("<d", expected) == struct.pack("<d", actual)
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for left, right in zip(expected, actual, strict=True):
            assert_json_equal(left, right)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert list(actual) == list(expected)
        for key, value in expected.items():
            assert_json_equal(value, actual[key])
    else:
        assert actual == expected
