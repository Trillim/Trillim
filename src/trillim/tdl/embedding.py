from __future__ import annotations

from .core import (
    Add,
    Cast,
    Compare,
    Constant,
    Construct,
    Gather,
    Mul,
    Reshape,
    Select,
    Sub,
    Tensor,
)
from .math import Clip, Take, Value, _dims, _tensor


@Construct
def Embedding(
    tokens: Tensor,
    weight: Value,
    *,
    token_rank: int = 1,
    padding_idx: int | None = None,
) -> Tensor:
    result = Gather(
        _tensor(weight), indices=Reshape(tokens, (*_dims(tokens, token_rank), 1))
    )
    if padding_idx is not None:
        mask = Reshape(
            Compare(tokens, padding_idx, predicate="eq"),
            (*_dims(tokens, token_rank), 1),
        )
        result = Select(mask, 0, result)
    return result


@Construct
def InterpolatedEmbedding(
    positions: Tensor, weight: Value, *, grid_size: int
) -> Tensor:
    """Bilinear learned square-grid positions. Coordinates [N,2] are in grid units."""

    y = Clip(
        Take(positions, Constant(0), axis=1, rank=2, index_rank=0), 0, grid_size - 1
    )
    x = Clip(
        Take(positions, Constant(1), axis=1, rank=2, index_rank=0), 0, grid_size - 1
    )
    y0, x0 = Cast(y, "int64"), Cast(x, "int64")
    dy, dx = Sub(y, y0), Sub(x, x0)
    result: Tensor | None = None
    for oy in (0, 1):
        for ox in (0, 1):
            yy, xx = (
                Clip(Add(y0, oy), 0, grid_size - 1),
                Clip(Add(x0, ox), 0, grid_size - 1),
            )
            value = Embedding(Add(Mul(yy, grid_size), xx), weight, token_rank=1)
            coefficient = Mul(
                dy if oy else Sub(Constant(1.0), dy),
                dx if ox else Sub(Constant(1.0), dx),
            )
            value = Mul(value, Reshape(coefficient, (-1, 1)))
            result = value if result is None else Add(result, value)
    assert result is not None
    return result
