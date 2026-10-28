from __future__ import annotations

from collections.abc import Callable

from .activation import Silu
from .core import Construct, Mul, Tensor
from .linear import Linear
from .math import Value


@Construct
def FFN(
    x: Tensor,
    *,
    up: Value,
    down: Value,
    gate: Value | None = None,
    up_bias: Value | None = None,
    down_bias: Value | None = None,
    gate_bias: Value | None = None,
    activation: Callable[[Tensor], Tensor] = Silu,
) -> Tensor:
    if gate is None and gate_bias is not None:
        raise ValueError("gate bias requires a gate projection")
    hidden = Linear(x, up, bias=up_bias)
    if gate is None:
        hidden = activation(hidden)
    else:
        hidden = Mul(hidden, activation(Linear(x, gate, bias=gate_bias)))
    return Linear(hidden, down, bias=down_bias)
