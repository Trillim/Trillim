from __future__ import annotations

from .core import Cache, Compare, Construct, Iota, Mul, Select, Sub, Tensor
from .math import Argmax, Dim, FullLike, Minimum, Slice, Take


@Construct
def ReusePrefix(
    tokens: Tensor,
    history: Cache,
    *,
    caches: tuple[Cache, ...],
    reset: Tensor | None = None,
) -> Tensor:
    """Reconcile a nonempty token prefix and return its prefill starting position.

    History is an axis-zero append cache recording only processed tokens. Append
    caches retain one entry per token; overwrite caches hold zero-initialized
    recurrent state. Rollback of recurrent state resets all caches for replay.
    Reset when conditioning changes; sliding caches cannot recover old prefixes.
    """
    if any(cache._spec.mode == "sliding" for cache in (history, *caches)):
        raise ValueError("prefix reuse requires caches that retain the full history")
    if (
        history._spec.mode != "append"
        or history._spec.axis != 0
        or len(history._spec.shape.dimensions) != 1
    ):
        raise ValueError("token history must be a rank-one axis-zero append cache")

    previous = history.read()
    length = Minimum(Dim(tokens, 0), Dim(previous, 0))
    indices = Iota(length)
    different = Compare(Take(tokens, indices), Take(previous, indices), predicate="ne")
    common = Minimum(Argmax(Mul(indices, -1), rank=1, mask=different), length)
    # Replay the final prompt token when its logits are not retained.
    start = Minimum(common, Sub(Dim(tokens, 0), 1))
    if any(cache._spec.mode == "overwrite" for cache in caches):
        start = Select(Compare(start, Dim(previous, 0), predicate="eq"), start, 0)
    if reset is not None:
        start = Select(reset, 0, start)

    history.assign(Slice(previous, axis=0, stop=start, rank=1))
    clear = Compare(start, 0, predicate="eq")
    for cache in caches:
        value = cache.read()
        axis = cache._spec.axis
        if axis is None:
            value = Select(clear, FullLike(value), value)
        else:
            value = Slice(
                value, axis=axis, stop=start, rank=len(cache._spec.shape.dimensions)
            )
        cache.assign(value)
    return start
