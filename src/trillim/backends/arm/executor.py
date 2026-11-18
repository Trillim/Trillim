import platform

from ...compiler.kernels.catalog import _available_capabilities
from ...compiler.kernels.native import _CAPABILITY_FLAGS
from ..cpu.executor import _CPU


class _ARM(_CPU):
    def _capabilities(self) -> int:
        if platform.machine().lower() not in {"aarch64", "arm64"}:
            raise ValueError("ARM backend requires an ARM64 device")
        return _available_capabilities("neon") | _CAPABILITY_FLAGS["neon"]["neon"]
