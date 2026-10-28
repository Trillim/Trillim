from __future__ import annotations

from .core import Add, Construct, Matmul, Permute, Tensor
from .math import Value, _tensor


@Construct
def Linear(
    x: Tensor,
    weight: Value,
    *,
    bias: Value | None = None,
) -> Tensor:
    """Apply [out, in] logical weights and optional bias."""
    output = Matmul(x, Permute(_tensor(weight), (1, 0)))
    if bias is not None:
        output = Add(output, _tensor(bias))
    return output
