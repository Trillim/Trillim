from __future__ import annotations

from .core import (
    Add,
    Cast,
    Compare,
    Construct,
    Div,
    Log,
    Mul,
    Random,
    Select,
    Shape,
    Sub,
    Tensor,
)
from .math import Argmax, CumSum, Dim, Maximum, Minimum, Softmax, Take, TopK


@Construct
def Greedy(logits: Tensor) -> Tensor:
    return Cast(Argmax(logits, rank=1), "int32")


@Construct
def Sample(
    logits: Tensor,
    *,
    seed: int | Tensor,
    offset: int | Tensor = 0,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
) -> Tensor:
    """Sample one logit vector using temperature, top-k and nucleus filtering."""
    if temperature <= 0 or not 0 < top_p <= 1 or (top_k is not None and top_k <= 0):
        raise ValueError("invalid sampling parameters")
    scores = Div(logits, temperature)
    indices: Tensor | None = None
    if top_k is not None or top_p < 1:
        scores, indices = TopK(
            scores,
            k=Dim(scores, -1) if top_k is None else Minimum(Dim(scores, -1), top_k),
            rank=1,
        )
    if top_p < 1:
        probabilities = Softmax(scores)
        before = Sub(CumSum(probabilities, rank=1), probabilities)
        scores = Select(Compare(before, top_p, predicate="lt"), scores, float("-inf"))
    uniform = Random(Shape(scores), seed=seed, offset=offset)
    gumbel = Mul(Log(Mul(Log(Maximum(uniform, 1e-30)), -1)), -1)
    chosen = Argmax(Add(scores, gumbel), rank=1)
    if indices is not None:
        chosen = Take(indices, chosen, index_rank=0)
    return Cast(chosen, "int32")
