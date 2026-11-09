from __future__ import annotations

import ctypes as c
from dataclasses import dataclass


@dataclass(slots=True)
class _Allocation:
    owner: c.Array[c.c_char]
    address: int
    size: int
