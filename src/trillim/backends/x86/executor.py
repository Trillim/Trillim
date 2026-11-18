import ctypes as c
import platform

from ...compiler.kernels.catalog import _available_capabilities
from ..cpu.executor import _CPU


class _X86(_CPU):
    def _capabilities(self) -> int:
        if platform.machine().lower() not in {"x86_64", "amd64", "i386", "i686", "x86"}:
            raise ValueError("x86 backend requires an x86 device")
        self.harness.trillim_cpu_features.argtypes = []
        self.harness.trillim_cpu_features.restype = c.c_uint64
        return _available_capabilities("x86") | int(self.harness.trillim_cpu_features())
