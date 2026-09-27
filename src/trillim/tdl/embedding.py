from __future__ import annotations

from typing import Literal

from .operator import _create_operator, _validate_variant
from .tdl_pydantic_models import embedding_args, op_args
from .tensor import Tensor

EmbeddingOp = Literal["embedding"]
EmbeddingArgs = embedding_args

_EMBEDDING_ARGS: dict[str, type[op_args]] = {
    "embedding": embedding_args,
}


def Embedding(*, op: EmbeddingOp, args: EmbeddingArgs) -> Tensor:
    _validate_variant(
        family="Embedding",
        op=op,
        args=args,
        registry=_EMBEDDING_ARGS,
    )
    return _create_operator(
        family="embedding",
        op=op,
        args=args,
    )
