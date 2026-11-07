from __future__ import annotations

import ctypes as c
from collections.abc import Mapping

from ..compiler_dataclasses import ShapeExpression
from ..kernels.kernel_dataclasses import ParameterSchema, ParameterValue
from ..kernels.native import NATIVE_TYPES
from .artifact_dataclasses import PackedParameters, Requirements
from .expressions import bound


def _checked_scalar(kind: str, value: float, name: str) -> int | float:
    if kind == "float":
        return float(value)
    native = NATIVE_TYPES[kind]
    bits = c.sizeof(native) * 8
    unsigned = kind.startswith("u") or kind == "size_t"
    minimum, maximum = (
        (0, 2**bits - 1) if unsigned else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    )
    if not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"kernel parameter out of range: {name}")
    return value


def pack(
    schema: ParameterSchema,
    parameters: Mapping[str, int | float | tuple[int | float, ...]],
) -> PackedParameters:
    value = schema.native_type()
    owners: list[object] = [value]
    if set(parameters) != {item.name for item in schema.fields}:
        raise ValueError("kernel parameter fields do not match the native schema")
    for item in schema.fields:
        argument = parameters[item.name]
        native = NATIVE_TYPES[item.type]
        if item.length is not None:
            if (
                not isinstance(argument, tuple)
                or len(argument) != parameters[item.length]
            ):
                raise ValueError(f"invalid parameter array: {item.name}")
            array = c.create_string_buffer(len(argument) * c.sizeof(native))
            for index, element in enumerate(argument):
                element = _checked_scalar(item.type, element, item.name)
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
            argument = _checked_scalar(item.type, argument, item.name)
            setattr(value, item.name, argument)
    return PackedParameters(c.addressof(value), owners)


def requirements(
    schema: ParameterSchema, parameters: Mapping[str, ParameterValue]
) -> Requirements:
    def scalar(value: float | ShapeExpression) -> int | float:
        return bound(value) if isinstance(value, ShapeExpression) else value

    packed = pack(
        schema,
        {
            name: tuple(scalar(item) for item in value)
            if isinstance(value, tuple)
            else scalar(value)
            for name, value in parameters.items()
        },
    )
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
    return Requirements(tuple(inputs), tuple(outputs), scratch.value)
