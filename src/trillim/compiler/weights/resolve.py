from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ...dtype import DType
from ...tdl.core.core_dataclasses import Shape
from ...tdl.core.core_pydantic_models import ConstantArgs, Weight
from ...tdl.core.graph import Graph
from ..passes.walk import walk
from .reader import LOGICAL_DTYPES, load_checkpoint
from .weights_dataclasses import Checkpoint, WeightBinding, WeightEntry, WeightIndex


def _entry(checkpoint: Checkpoint, name: str) -> WeightEntry:
    canonical = name
    visited = set()
    while canonical in checkpoint.aliases:
        if canonical in visited:
            raise ValueError(f"cyclic checkpoint weight aliases: {name}")
        visited.add(canonical)
        canonical = checkpoint.aliases[canonical]
    aliases = tuple(
        f"{checkpoint.name}.{alias}"
        for alias, target in checkpoint.aliases.items()
        if target == canonical
    )
    source = checkpoint.tensors.get(canonical)
    if source is None:
        stem = canonical.removesuffix(".weight")
        blocks = checkpoint.tensors.get(stem + "_blocks")
        scales = checkpoint.tensors.get(stem + "_scales")
        if blocks is None or scales is None:
            raise ValueError(f"checkpoint {checkpoint.name!r} has no tensor {name!r}")
        if (
            blocks.dtype != "U8"
            or scales.dtype != "U8"
            or len(blocks.shape) < 2
            or blocks.shape[-1] != 16
            or blocks.shape[:-1] != scales.shape
        ):
            raise ValueError(f"invalid packed MXFP4 checkpoint tensors: {name}")
        shape = (*blocks.shape[:-2], blocks.shape[-2] * 32)
        return WeightEntry(
            f"{checkpoint.name}.{canonical}",
            shape,
            "mxfp4",
            blocks,
            "mxfp4",
            (blocks, scales),
            aliases,
        )
    tied = tuple(
        checkpoint.tensors[alias]
        for alias, target in checkpoint.aliases.items()
        if target == canonical and alias in checkpoint.tensors
    )
    if any(
        other.shape != source.shape or other.dtype != source.dtype for other in tied
    ):
        raise ValueError(f"tied checkpoint tensors disagree in shape or dtype: {name}")
    quantization = checkpoint.config.quantization_config
    bitnet = (
        checkpoint.config.model_type == "bitnet"
        or any(
            "bitnet" in architecture.lower()
            for architecture in checkpoint.config.architectures
        )
        or quantization is not None
        and quantization.quant_method == "bitnet"
    )
    if bitnet and canonical.endswith("_proj.weight"):
        if len(source.shape) != 2:
            raise ValueError(f"BitNet projection must be a matrix: {name}")
        if source.dtype == "U8":
            scale_name = canonical.removesuffix("weight") + "weight_scale"
            scale = checkpoint.tensors.get(scale_name)
            if (
                quantization is None
                or quantization.quantization_mode != "offline"
                or quantization.linear_class not in ("bitlinear", "autobitlinear")
                or scale is None
            ):
                raise ValueError(
                    f"packed BitNet projection needs offline configuration and scalar weight_scale: {name}"
                )
            return WeightEntry(
                f"{checkpoint.name}.{canonical}",
                (source.shape[0] * 4, source.shape[1]),
                "mat_ternary",
                source,
                "bitnet_packed",
                (scale,),
                aliases,
                tied,
                quantization.linear_class,
            )
        if source.dtype not in ("BF16", "F16", "F32"):
            raise ValueError(f"unsupported BitNet latent weight dtype: {source.dtype}")
        return WeightEntry(
            f"{checkpoint.name}.{canonical}",
            source.shape,
            "mat_ternary",
            source,
            "bitnet_absmean",
            (),
            aliases,
            tied,
        )
    return WeightEntry(
        f"{checkpoint.name}.{canonical}",
        source.shape,
        LOGICAL_DTYPES[source.dtype],
        source,
        aliases=aliases,
        tied_sources=tied,
    )


def resolve_weights(
    graph: Graph,
    checkpoints: Mapping[str, Path],
    *,
    dtypes: Mapping[str, DType] | None = None,
) -> WeightIndex:
    """Bind checkpoint-qualified logical identities before shape inference."""
    if graph.region is None:
        raise ValueError("weight resolution requires a completed graph")
    overrides = dtypes or {}
    index = WeightIndex()
    identities: dict[str, WeightEntry] = {}
    for _, operator in walk(graph.region):
        args = operator.args
        if not isinstance(args, ConstantArgs) or not isinstance(args.value, Weight):
            continue
        weight = args.value
        checkpoint = index.checkpoints.get(weight.model_name)
        if checkpoint is None:
            location = checkpoints.get(weight.model_name)
            if location is None:
                raise ValueError(f"missing checkpoint binding: {weight.model_name}")
            checkpoint = load_checkpoint(weight.model_name, location)
            index.checkpoints[weight.model_name] = checkpoint
        source = _entry(checkpoint, weight.tensor_name)
        source = identities.setdefault(source.identity, source)
        if weight.shape is not None and (
            len(weight.shape.dimensions) != len(source.shape)
            or any(
                declared is not None and declared != actual
                for declared, actual in zip(weight.shape.dimensions, source.shape)
            )
        ):
            raise ValueError(
                f"declared weight shape disagrees with checkpoint: {weight.root}"
            )
        target = overrides.get(weight.root) or weight.dtype or args.dtype
        if (
            target is None
            and len(source.shape) >= 2
            and source.dtype
            not in (
                "bool",
                "int8",
                "uint8",
                "int16",
                "uint16",
                "int32",
                "uint32",
                "int64",
                "uint64",
            )
        ):
            target = overrides.get(weight.model_name)
        target = target or source.dtype
        weight.shape = Shape(source.shape)
        operator.args = args.model_copy(update={"dtype": target})
        for tensor in operator.outputs:
            tensor.shape, tensor.dtype = weight.shape, target
            index.bindings[tensor] = WeightBinding(weight.root, source, target)
        index.sources[weight.root] = source
        index.sources.setdefault(source.identity, source)
        for alias in source.aliases:
            index.sources.setdefault(alias, source)
    return index
