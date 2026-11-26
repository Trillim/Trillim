from __future__ import annotations

from typing import Literal, cast, get_args

DType = Literal[
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "fp16",
    "bf16",
    "fp32",
    "row_int8",
    "mat_ternary",
    "grp_int8",
    "grp_int4",
    "grp_int5",
    "dubgrp_int6",
    "grp_binary",
    "grp_ternary",
    "mxfp4",
    "fp8_e4m3",
]


def _validate_dtype(value: str) -> DType:
    if value not in get_args(DType):
        raise ValueError(f'unsupported dtype "{value}"')
    return cast(DType, value)
