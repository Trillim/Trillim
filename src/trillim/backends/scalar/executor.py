from ...compiler.kernels.catalog import _available_capabilities
from ..cpu.executor import _CPU


class _Scalar(_CPU):
    def _capabilities(self) -> int:
        return _available_capabilities("scalar")
