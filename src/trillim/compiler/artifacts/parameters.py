from __future__ import annotations

import ctypes as c
from collections.abc import Mapping
from typing import cast

from ..compiler_dataclasses import _ShapeExpression
from ..kernels.kernel_dataclasses import _ParameterSchema, _ParameterValue
from ..kernels.native import _NATIVE_TYPES
from .artifact_dataclasses import _PackedParameters, _Requirements
from .expressions import _bound


def _checked_scalar(kind: str, value: float, name: str) -> int | float:
    if kind == "float":
        return float(value)
    native = _NATIVE_TYPES[kind]
    bits = c.sizeof(native) * 8
    unsigned = kind.startswith("u") or kind == "size_t"
    minimum, maximum = (
        (0, 2**bits - 1) if unsigned else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    )
    if not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"kernel parameter out of range: {name}")
    return value


def _pack(
    schema: _ParameterSchema,
    parameters: Mapping[str, _ParameterValue],
    *,
    previous: tuple[_PackedParameters, Mapping[str, _ParameterValue]] | None = None,
) -> _PackedParameters:
    """Reuse previous storage only after its native invocation has drained."""

    def scalar(value: float | _ShapeExpression) -> int | float:
        return _bound(value) if isinstance(value, _ShapeExpression) else value

    value = (
        cast(c.Structure, previous[0].owners[0]) if previous else schema.native_type()
    )
    owners: list[object] = [value]
    owner_index = 1
    if set(parameters) != {item.name for item in schema.fields}:
        raise ValueError("kernel parameter fields do not match the native schema")
    for item in schema.fields:
        argument = parameters[item.name]
        native = _NATIVE_TYPES[item.type]
        if item.length is not None:
            length = parameters[item.length]
            if (
                not isinstance(argument, tuple)
                or isinstance(length, tuple)
                or len(argument) != scalar(length)
            ):
                raise ValueError(f"invalid parameter array: {item.name}")
            if previous is not None and parameters[item.name] == previous[1][item.name]:
                owners.append(previous[0].owners[owner_index])
                owner_index += 1
                continue
            owner_index += 1
            array = c.create_string_buffer(len(argument) * c.sizeof(native))
            for index, element in enumerate(argument):
                element = _checked_scalar(item.type, scalar(element), item.name)
                converted = (
                    c.c_float(float(element))
                    if item.type == "float"
                    else native(int(element))
                )
                c.memmove(
                    c.addressof(array) + index * c.sizeof(native),
                    c.byref(converted),
                    c.sizeof(native),
                )
            owners.append(array)
            setattr(value, item.name, c.addressof(array))
        else:
            if isinstance(argument, tuple):
                raise ValueError(f"expected scalar parameter: {item.name}")
            if previous is not None and parameters[item.name] == previous[1][item.name]:
                continue
            argument = _checked_scalar(item.type, scalar(argument), item.name)
            setattr(value, item.name, argument)
    return _PackedParameters(c.addressof(value), owners)


def _requirements(
    schema: _ParameterSchema,
    parameters: Mapping[str, _ParameterValue],
    *,
    packed: _PackedParameters | None = None,
) -> _Requirements:
    if packed is None:
        packed = _pack(schema, parameters)
    inputs = (c.c_size_t * schema.input_count)()
    outputs = (c.c_size_t * schema.output_count)()
    scratch = c.c_size_t()
    function = c.CFUNCTYPE(
        c.c_int32,
        c.c_void_p,
        c.POINTER(c.c_size_t),
        c.POINTER(c.c_size_t),
        c.POINTER(c.c_size_t),
    )(schema.requirements_address)
    status = function(packed.address, inputs, outputs, c.byref(scratch))
    if status:
        raise ValueError(f"kernel rejected bounded parameters (status {status})")
    return _Requirements(tuple(inputs), tuple(outputs), scratch.value)


def _can_alias(
    schema: _ParameterSchema,
    parameters: Mapping[str, _ParameterValue],
    output: int,
    input: int,
) -> bool:
    if schema.can_alias_address is None:
        return False
    dynamic = any(
        isinstance(item, _ShapeExpression) and item.op != "const"
        for value in parameters.values()
        for item in (value if isinstance(value, tuple) else (value,))
    )
    packed = None if dynamic else _pack(schema, parameters)
    function = c.CFUNCTYPE(c.c_bool, c.c_void_p, c.c_size_t, c.c_size_t)(
        schema.can_alias_address
    )
    return bool(function(None if packed is None else packed.address, output, input))
