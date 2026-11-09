from __future__ import annotations

import ctypes as c
from typing import cast

from .kernel_dataclasses import _Backend, _ParameterField, _ParameterSchema

_SOFTWARE_CAPABILITIES = {"m": 1 << 32, "blas": 1 << 33}
_CAPABILITY_FLAGS: dict[_Backend, dict[str, int]] = {
    "scalar": {**_SOFTWARE_CAPABILITIES},
    "x86": {
        **_SOFTWARE_CAPABILITIES,
        "avx2": 1 << 0,
        "avx512f": 1 << 1,
        "avx512bw": 1 << 2,
        "avx512dq": 1 << 3,
        "avx512vl": 1 << 4,
        "avx_vnni": 1 << 5,
        "avx512_vnni": 1 << 6,
        "fma": 1 << 7,
    },
    "neon": {
        **_SOFTWARE_CAPABILITIES,
        "neon": 1 << 0,
        "dotprod": 1 << 1,
    },
}


class _NativeParameter(c.Structure):
    _fields_ = [
        ("name", c.c_char_p),
        ("type", c.c_char_p),
        ("offset", c.c_size_t),
        ("length", c.c_char_p),
    ]


class _NativeSchema(c.Structure):
    _fields_ = [
        ("size", c.c_size_t),
        ("alignment", c.c_size_t),
        ("field_count", c.c_size_t),
        ("fields", c.POINTER(_NativeParameter)),
        ("input_count", c.c_size_t),
        ("output_count", c.c_size_t),
        ("requirements", c.c_void_p),
        ("can_alias", c.c_void_p),
        ("input_alignments", c.POINTER(c.c_size_t)),
        ("output_alignments", c.POINTER(c.c_size_t)),
        ("scratch_alignment", c.c_size_t),
    ]


class _NativeCapabilities(c.Structure):
    _fields_ = [
        ("abi_version", c.c_uint32),
        ("required", c.c_uint64),
    ]


_NATIVE_TYPES: dict[
    str, type[c.c_size_t | c.c_int32 | c.c_uint32 | c.c_int64 | c.c_uint64 | c.c_float]
] = {
    "size_t": c.c_size_t,
    "int32_t": c.c_int32,
    "uint32_t": c.c_uint32,
    "int64_t": c.c_int64,
    "uint64_t": c.c_uint64,
    "float": c.c_float,
}


def _read_capabilities(library: c.CDLL, name: str) -> int:
    function = getattr(library, f"{name}_capabilities")
    function.argtypes, function.restype = [], c.POINTER(_NativeCapabilities)
    capabilities = cast("c._Pointer[_NativeCapabilities]", function()).contents
    if capabilities.abi_version != 1:
        raise ValueError(f"{name}: incompatible kernel ABI")
    return int(capabilities.required)


def _read_schema(library: c.CDLL, name: str) -> _ParameterSchema:
    native_function = getattr(library, f"{name}_parameters")
    native_function.argtypes, native_function.restype = (
        [],
        c.POINTER(_NativeSchema),
    )
    native = cast("c._Pointer[_NativeSchema]", native_function()).contents
    fields = tuple(
        _ParameterField(
            field.name.decode(),
            field.type.decode(),
            field.offset,
            field.length.decode() if field.length else None,
        )
        for field in native.fields[: native.field_count]
    )
    native_type = cast(
        type[c.Structure],
        type(
            f"{name}Parameters",
            (c.Structure,),
            {
                "_fields_": [
                    (
                        field.name,
                        c.c_void_p if field.length else _NATIVE_TYPES[field.type],
                    )
                    for field in fields
                ]
            },
        ),
    )
    if (
        c.sizeof(native_type) != native.size
        or c.alignment(native_type) != native.alignment
    ):
        raise ValueError(f"{name}: inconsistent parameter schema")
    for field in fields:
        if getattr(native_type, field.name).offset != field.offset:
            raise ValueError(f"{name}: inconsistent field offset")
    return _ParameterSchema(
        native.size,
        native.alignment,
        fields,
        native.input_count,
        native.output_count,
        tuple(native.input_alignments[: native.input_count])
        if native.input_count
        else (),
        tuple(native.output_alignments[: native.output_count])
        if native.output_count
        else (),
        native.scratch_alignment,
        int(native.requirements),
        native_type,
    )
