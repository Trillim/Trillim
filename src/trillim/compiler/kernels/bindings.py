from __future__ import annotations

from typing import get_args

from ...tdl.core.tensor import Tensor
from ...tdl.core.types import DType
from ..compiler_dataclasses import ShapeExpression
from ..passes.infer import constant, expression
from .kernel_dataclasses import KernelBindingContext, ParameterValue, TensorBinding

DTYPE_IDS: dict[DType, int] = {
    name: index for index, name in enumerate(get_args(DType))
}


def product(shape: tuple[ShapeExpression, ...]) -> ShapeExpression:
    result = constant(1)
    for dimension in shape:
        result = expression("mul", result, dimension)
    return result


def contiguous_strides(
    shape: tuple[ShapeExpression, ...],
) -> tuple[ShapeExpression, ...]:
    strides: list[ShapeExpression] = []
    size = constant(1)
    for dimension in reversed(shape):
        strides.append(size)
        size = expression("mul", size, dimension)
    return tuple(reversed(strides))


def broadcast_strides(
    shape: tuple[ShapeExpression, ...],
    target: tuple[ShapeExpression, ...],
    strides: tuple[ShapeExpression, ...] | None = None,
) -> tuple[ShapeExpression, ...]:
    if len(shape) > len(target):
        raise ValueError("cannot broadcast to a smaller rank")
    strides = strides or contiguous_strides(shape)
    result = [constant(0)] * (len(target) - len(shape))
    for dimension, stride, output in zip(
        shape, strides, target[len(target) - len(shape) :], strict=True
    ):
        if dimension.op == "const" and dimension.value == 1:
            result.append(constant(0))
        elif dimension == output:
            result.append(stride)
        else:
            result.append(expression("broadcast_stride", dimension, stride))
    return tuple(result)


def native_dtype(dtype: DType) -> int:
    result = DTYPE_IDS[dtype]
    if result > DTYPE_IDS["fp32"]:
        raise ValueError("kernel requires native tensor storage")
    return result


def binary(context: KernelBindingContext) -> dict[str, ParameterValue]:
    output = context.outputs[0]
    a, b = context.inputs["a"], context.inputs["b"]
    if a.dtype != output.dtype or b.dtype != output.dtype:
        raise ValueError("binary inputs must have the promoted output dtype")
    return {
        "count": product(output.shape),
        "rank": len(output.shape),
        "shape": output.shape,
        "a_strides": broadcast_strides(a.shape, output.shape, a.strides),
        "b_strides": broadcast_strides(b.shape, output.shape, b.strides),
        "dtype": native_dtype(output.dtype),
    }


def unary(context: KernelBindingContext) -> dict[str, ParameterValue]:
    output = context.outputs[0]
    source = context.inputs["x"]
    return {
        "count": product(output.shape),
        "rank": len(output.shape),
        "shape": output.shape,
        "strides": source.strides or contiguous_strides(source.shape),
        "dtype": native_dtype(output.dtype),
    }


def storage_shape(binding: TensorBinding) -> tuple[ShapeExpression, ShapeExpression]:
    if binding.storage_shape is not None:
        return binding.storage_shape
    if len(binding.shape) < 2:
        return constant(1), product(binding.shape)
    return binding.shape[0], product(binding.shape[1:])


def _batch_strides(
    binding: TensorBinding, target: tuple[ShapeExpression, ...]
) -> tuple[ShapeExpression, ...]:
    shape = binding.shape[:-2]
    strides = (binding.strides or contiguous_strides(binding.shape))[:-2]
    result = [constant(0)] * (len(target) - len(shape))
    for dimension, stride, output in zip(
        shape, strides, target[len(target) - len(shape) :], strict=True
    ):
        if dimension.op == "const" and dimension.value == 1:
            result.append(constant(0))
        elif dimension == output:
            result.append(stride)
        else:
            result.append(expression("broadcast_stride", dimension, stride))
    return tuple(result)


def matmul(context: KernelBindingContext) -> dict[str, ParameterValue]:
    a, b = context.inputs["a"], context.inputs["b"]
    output = context.outputs[0]
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError("matrix multiplication requires rank at least two")
    if (not output.dtype.startswith("fp") and output.dtype != "bf16") and (
        DTYPE_IDS[a.dtype] >= DTYPE_IDS["fp16"]
        or DTYPE_IDS[b.dtype] >= DTYPE_IDS["fp16"]
    ):
        raise ValueError("floating matrix inputs require a floating output")
    batch = output.shape[:-2]
    a_strides = a.strides or contiguous_strides(a.shape)
    b_strides = b.strides or contiguous_strides(b.shape)
    a_rows, a_columns = storage_shape(a)
    b_rows, b_columns = storage_shape(b)
    return {
        "batches": product(batch),
        "m": output.shape[-2],
        "n": output.shape[-1],
        "k": a.shape[-1],
        "rank": len(batch),
        "batch_shape": batch,
        "a_batch_strides": _batch_strides(a, batch),
        "b_batch_strides": _batch_strides(b, batch),
        "a_row_stride": a_strides[-2],
        "a_col_stride": a_strides[-1],
        "b_row_stride": b_strides[-2],
        "b_col_stride": b_strides[-1],
        "a_dtype": DTYPE_IDS[a.dtype],
        "b_dtype": DTYPE_IDS[b.dtype],
        "out_dtype": native_dtype(output.dtype),
        "a_storage_rows": a_rows,
        "a_storage_columns": a_columns,
        "b_storage_rows": b_rows,
        "b_storage_columns": b_columns,
    }


def cast(context: KernelBindingContext) -> dict[str, ParameterValue]:
    source = context.inputs["x"]
    output = context.outputs[0]
    source_rows, source_columns = storage_shape(source)
    output_rows, output_columns = storage_shape(output)
    return {
        "count": product(output.shape),
        "rank": len(source.shape),
        "shape": source.shape,
        "strides": source.strides or contiguous_strides(source.shape),
        "source_offset": 0,
        "source_rows": source_rows,
        "source_columns": source_columns,
        "output_rows": output_rows,
        "output_columns": output_columns,
        "input_dtype": DTYPE_IDS[source.dtype],
        "output_dtype": DTYPE_IDS[output.dtype],
    }


def _reduction(
    context: KernelBindingContext, axes: tuple[int, ...]
) -> dict[str, ParameterValue]:
    source = context.inputs["x"]
    rank = len(source.shape)
    normalized = tuple(axis % rank for axis in axes)
    strides = source.strides or contiguous_strides(source.shape)
    kept = tuple(axis for axis in range(rank) if axis not in normalized)
    rows, columns = storage_shape(source)
    return {
        "outputs": product(context.outputs[0].shape),
        "output_rank": len(kept),
        "reduce_rank": len(normalized),
        "reduce_count": product(tuple(source.shape[axis] for axis in normalized)),
        "output_shape": tuple(source.shape[axis] for axis in kept),
        "output_strides": tuple(strides[axis] for axis in kept),
        "reduce_shape": tuple(source.shape[axis] for axis in normalized),
        "reduce_strides": tuple(strides[axis] for axis in normalized),
        "storage_rows": rows,
        "storage_columns": columns,
        "input_dtype": DTYPE_IDS[source.dtype],
        "output_dtype": native_dtype(context.outputs[0].dtype),
    }


def reduction(context: KernelBindingContext) -> dict[str, ParameterValue]:
    axes = context.attrs["axes"]
    if not isinstance(axes, tuple) or not all(isinstance(axis, int) for axis in axes):
        raise TypeError("reduction axes must be an integer tuple")
    return _reduction(context, axes)


def argmax(context: KernelBindingContext) -> dict[str, ParameterValue]:
    axis = context.attrs.get("axis", -1)
    if not isinstance(axis, int):
        raise TypeError("argmax axis must be an integer")
    return _reduction(context, (axis,))


def compare(context: KernelBindingContext) -> dict[str, ParameterValue]:
    output = context.outputs[0]
    dtype = context.inputs["a"].dtype
    promoted = TensorBinding(output.shape, output.maximum_shape, dtype)
    result = binary(KernelBindingContext(context.inputs, (promoted,), context.attrs))
    predicate = context.attrs["predicate"]
    if not isinstance(predicate, str) or predicate not in (
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
    ):
        raise ValueError("invalid comparison predicate")
    result["predicate"] = ("eq", "ne", "lt", "le", "gt", "ge").index(predicate)
    return result


def select(context: KernelBindingContext) -> dict[str, ParameterValue]:
    result = binary(
        KernelBindingContext(
            {"a": context.inputs["when_true"], "b": context.inputs["when_false"]},
            context.outputs,
            context.attrs,
        )
    )
    condition = context.inputs["condition"]
    result["condition_strides"] = broadcast_strides(
        condition.shape, context.outputs[0].shape, condition.strides
    )
    return result


def permute(context: KernelBindingContext) -> dict[str, ParameterValue]:
    result = cast(context)
    axes = context.attrs["axes"]
    if not isinstance(axes, tuple) or not all(isinstance(axis, int) for axis in axes):
        raise TypeError("permute axes must be an integer tuple")
    source = context.inputs["x"]
    strides = source.strides or contiguous_strides(source.shape)
    result.update(
        {
            "rank": len(context.outputs[0].shape),
            "shape": context.outputs[0].shape,
            "strides": tuple(strides[axis] for axis in axes),
        }
    )
    return result


def broadcast(context: KernelBindingContext) -> dict[str, ParameterValue]:
    result = cast(context)
    source = context.inputs["x"]
    output = context.outputs[0]
    result.update(
        {
            "rank": len(output.shape),
            "shape": output.shape,
            "strides": broadcast_strides(source.shape, output.shape, source.strides),
        }
    )
    return result


def shape(context: KernelBindingContext) -> dict[str, ParameterValue]:
    source = context.inputs["x"]
    return {"rank": len(source.shape), "shape": source.shape}


def _scalar_parameter(
    value: object, binding: TensorBinding | None = None
) -> int | ShapeExpression:
    if isinstance(value, (int, ShapeExpression)):
        return value
    if isinstance(value, Tensor):
        if (
            binding is not None
            and binding.values is not None
            and len(binding.values) == 1
        ):
            return binding.values[0]
        raise ValueError(
            "scalar tensor kernel parameters require inferred finite bounds"
        )
    raise TypeError("expected an integer or scalar tensor parameter")


def iota(context: KernelBindingContext) -> dict[str, ParameterValue]:
    return {
        "count": product(context.outputs[0].shape),
        "start": _scalar_parameter(
            context.attrs.get("start", 0), context.inputs.get("start")
        ),
        "step": _scalar_parameter(context.attrs.get("step", 1)),
        "dtype": native_dtype(context.outputs[0].dtype),
    }


def random(context: KernelBindingContext) -> dict[str, ParameterValue]:
    distribution = context.attrs.get("distribution", "uniform")
    if distribution not in ("uniform", "normal"):
        raise ValueError("invalid random distribution")
    return {
        "count": product(context.outputs[0].shape),
        "seed": _scalar_parameter(context.attrs["seed"], context.inputs.get("seed")),
        "offset": _scalar_parameter(
            context.attrs.get("offset", 0), context.inputs.get("offset")
        ),
        "distribution": int(distribution == "normal"),
        "dtype": native_dtype(context.outputs[0].dtype),
    }


def _indexing(
    context: KernelBindingContext, scatter_operation: bool
) -> dict[str, ParameterValue]:
    source = context.inputs["x"]
    indices = context.inputs["indices"]
    output = context.outputs[0]
    coordinates = indices.shape[-1].value
    if coordinates is None or coordinates < 0 or coordinates > len(source.shape):
        raise ValueError("indexed coordinate width must be statically known")
    if indices.dtype != "int64":
        raise ValueError("index kernels require promoted int64 coordinates")
    source_strides = source.strides or contiguous_strides(source.shape)
    index_strides = indices.strides or contiguous_strides(indices.shape)
    index_shape = indices.shape[:-1]
    slice_shape = source.shape[coordinates:]
    target_shape = (*index_shape, *slice_shape)
    storage_rows, storage_columns = storage_shape(source)
    output_rows, output_columns = storage_shape(output)
    mask = context.inputs.get("mask")
    fill = context.attrs.get("fill", 0)
    if not isinstance(fill, (int, float)):
        raise TypeError("gather fill must be a numeric scalar")
    mode = context.attrs.get("mode", "replace")
    if mode not in ("replace", "add"):
        raise ValueError("invalid scatter mode")
    if scatter_operation:
        native_dtype(output.dtype)
    floating = (
        output.dtype in ("fp16", "bf16", "fp32")
        or DTYPE_IDS[output.dtype] >= DTYPE_IDS["row_int8"]
    )
    values = context.inputs.get("values")
    return {
        "count": product(output.shape),
        "items": product(index_shape),
        "coordinates": coordinates,
        "index_rank": len(index_shape),
        "slice_rank": len(slice_shape),
        "slice_size": product(slice_shape),
        "index_shape": index_shape,
        "index_strides": index_strides[:-1],
        "mask_strides": broadcast_strides(mask.shape, index_shape, mask.strides)
        if mask
        else (constant(0),) * len(index_shape),
        "coordinate_stride": index_strides[-1],
        "indexed_shape": source.shape[:coordinates],
        "indexed_strides": source_strides[:coordinates],
        "output_indexed_strides": contiguous_strides(source.shape)[:coordinates],
        "slice_shape": slice_shape,
        "slice_strides": source_strides[coordinates:],
        "values_rank": len(target_shape) if values else 0,
        "values_shape": target_shape if values else (),
        "values_strides": broadcast_strides(values.shape, target_shape, values.strides)
        if values
        else (),
        "storage_rows": storage_rows,
        "storage_columns": storage_columns,
        "output_rows": output_rows,
        "output_columns": output_columns,
        "dtype": DTYPE_IDS[output.dtype],
        "masked": int(mask is not None),
        "mode": int(mode == "add"),
        "fill_bits": 0
        if floating
        else ((int(bool(fill)) if output.dtype == "bool" else int(fill)) & (2**64 - 1)),
        "fill_float": float(fill),
    }


def gather(context: KernelBindingContext) -> dict[str, ParameterValue]:
    return _indexing(context, False)


def scatter(context: KernelBindingContext) -> dict[str, ParameterValue]:
    return _indexing(context, True)


def reshape_strides(
    shape: tuple[ShapeExpression, ...],
    strides: tuple[ShapeExpression, ...],
    target: tuple[ShapeExpression, ...],
) -> tuple[ShapeExpression, ...] | None:
    if len(shape) != len(strides):
        raise ValueError("tensor shape and strides must have equal rank")
    expected = constant(1)
    contiguous = True
    for dimension, stride in zip(reversed(shape), reversed(strides), strict=True):
        if dimension.op != "const" or dimension.value != 1:
            contiguous = contiguous and stride == expected
        expected = expression("mul", expected, dimension)
    if contiguous:
        return contiguous_strides(target)
    source_axes = tuple(
        (dimension, stride)
        for dimension, stride in zip(shape, strides, strict=True)
        if dimension.op != "const" or dimension.value != 1
    )
    target_axes = tuple(
        dimension
        for dimension in target
        if dimension.op != "const" or dimension.value != 1
    )
    if tuple(dimension for dimension, _ in source_axes) != target_axes:
        return None
    iterator = iter(source_axes)
    return tuple(
        constant(0)
        if dimension.op == "const" and dimension.value == 1
        else next(iterator)[1]
        for dimension in target
    )


def rmsnorm(context: KernelBindingContext) -> dict[str, ParameterValue]:
    source = context.inputs["x"]
    if not source.shape:
        raise ValueError("RMSNorm requires a channel axis")
    source_strides = source.strides or contiguous_strides(source.shape)
    if reshape_strides(
        source.shape, source_strides, source.shape
    ) != contiguous_strides(source.shape):
        raise ValueError("fused RMSNorm requires contiguous input")
    rows, columns = storage_shape(source)
    weight = context.inputs.get("weight")
    if weight is not None and (
        len(weight.shape) != 1 or weight.shape[0] != source.shape[-1]
    ):
        raise ValueError(
            "fused RMSNorm requires a vector weight matching the channel axis"
        )
    eps = context.attrs.get("eps", 1e-6)
    offset = context.attrs.get("weight_offset", 0.0)
    if not isinstance(eps, (int, float)) or not isinstance(offset, (int, float)):
        raise TypeError("RMSNorm eps and weight_offset must be numeric scalars")
    return {
        "rows": product(source.shape[:-1]),
        "width": source.shape[-1],
        "storage_rows": rows,
        "storage_columns": columns,
        "input_dtype": DTYPE_IDS[source.dtype],
        "weight_dtype": DTYPE_IDS[weight.dtype] if weight else DTYPE_IDS["fp32"],
        "output_dtype": native_dtype(context.outputs[0].dtype),
        "eps": float(eps),
        "weight_offset": float(offset),
    }


def linear(context: KernelBindingContext) -> dict[str, ParameterValue]:
    source = context.inputs["x"]
    weight = context.inputs["weight"]
    output = context.outputs[0]
    if not source.shape or len(weight.shape) != 2:
        raise ValueError("linear requires a channel axis and matrix weights")
    rows = product(source.shape[:-1])
    source_shape = (rows, source.shape[-1])
    source_strides = reshape_strides(
        source.shape, source.strides or contiguous_strides(source.shape), source_shape
    )
    if source_strides is None:
        raise ValueError("fused linear requires a matrix-representable input view")
    matrix = TensorBinding(
        source_shape,
        (rows.maximum, source.maximum_shape[-1]),
        source.dtype,
        source_strides,
        storage_shape(source),
    )
    strides = weight.strides or contiguous_strides(weight.shape)
    transposed = TensorBinding(
        (weight.shape[1], weight.shape[0]),
        (weight.maximum_shape[1], weight.maximum_shape[0]),
        weight.dtype,
        (strides[1], strides[0]),
        storage_shape(weight),
    )
    result = TensorBinding(
        (rows, weight.shape[0]), (rows.maximum, weight.maximum_shape[0]), output.dtype
    )
    return matmul(
        KernelBindingContext({"a": matrix, "b": transposed}, (result,), context.attrs)
    )


def bitlinear(context: KernelBindingContext) -> dict[str, ParameterValue]:
    source = context.inputs["x"]
    weight = context.inputs["weight"]
    if (
        source.dtype != "fp32"
        or weight.dtype != "mat_ternary"
        or context.outputs[0].dtype != "fp32"
    ):
        raise ValueError(
            "fused BitLinear requires FP32 activations and canonical matrix-ternary weights"
        )
    if len(weight.shape) != 2 or not source.shape:
        raise ValueError("fused BitLinear requires matrix weights")
    for binding in (source, weight):
        strides = binding.strides or contiguous_strides(binding.shape)
        if reshape_strides(binding.shape, strides, binding.shape) != contiguous_strides(
            binding.shape
        ):
            raise ValueError("fused BitLinear requires contiguous operands")
    return {
        "tokens": product(source.shape[:-1]),
        "width": source.shape[-1],
        "outputs": weight.shape[0],
    }


def avx2_bitlinear(context: KernelBindingContext) -> dict[str, ParameterValue]:
    result = bitlinear(context)
    width = context.inputs["x"].shape[-1]
    if width.op != "const" or width.value is None or width.value % 128:
        raise ValueError("AVX2 BitLinear requires a fixed width divisible by 128")
    return result


def avx2_linear(context: KernelBindingContext) -> dict[str, ParameterValue]:
    result = linear(context)
    if context.inputs["x"].dtype != "fp32" or context.outputs[0].dtype != "fp32":
        raise ValueError("AVX2 linear requires FP32 activations and output")
    for name in ("a_col_stride", "b_row_stride"):
        stride = result[name]
        if (
            not isinstance(stride, ShapeExpression)
            or stride.op != "const"
            or stride.value != 1
        ):
            raise ValueError("AVX2 linear requires contiguous contraction axes")
    return result


def avx2_rmsnorm(context: KernelBindingContext) -> dict[str, ParameterValue]:
    result = rmsnorm(context)
    if context.inputs["x"].dtype != "fp32" or context.outputs[0].dtype != "fp32":
        raise ValueError("AVX2 RMSNorm requires FP32 activations and output")
    return result


def activation(context: KernelBindingContext) -> dict[str, ParameterValue]:
    output = context.outputs[0]
    for name in ("a", "b"):
        source = context.inputs[name]
        strides = source.strides or contiguous_strides(source.shape)
        if (
            source.dtype != "fp32"
            or source.shape != output.shape
            or strides != contiguous_strides(source.shape)
        ):
            raise ValueError(
                "fused gate activation requires matching contiguous FP32 operands"
            )
    if output.dtype != "fp32":
        raise ValueError("fused gate activation requires FP32 output")
    return {"count": product(output.shape)}


def rotary(context: KernelBindingContext) -> dict[str, ParameterValue]:
    import math

    source = context.inputs["x"]
    positions = context.inputs["positions"]
    if (
        len(source.shape) != 3
        or len(positions.shape) != 1
        or source.dtype != "fp32"
        or context.outputs[0].dtype != "fp32"
    ):
        raise ValueError(
            "fused Rotary requires FP32 [heads,tokens,channels] and one-dimensional positions"
        )
    dim = context.attrs["dim"]
    theta = context.attrs.get("theta", 10000.0)
    amplitude = context.attrs.get("amplitude", 1.0)
    interleaved = context.attrs.get("interleaved", False)
    if not isinstance(dim, int) or dim <= 0 or dim % 2:
        raise ValueError("rotary dim must be positive and even")
    if (
        not isinstance(theta, (int, float))
        or theta <= 0
        or not isinstance(amplitude, (int, float))
    ):
        raise ValueError("invalid rotary frequency or amplitude")
    if not isinstance(interleaved, bool):
        raise TypeError("interleaved must be boolean")
    return {
        "heads": source.shape[0],
        "tokens": source.shape[1],
        "dim": source.shape[2],
        "rotary_dim": dim,
        "rank": 3,
        "strides": source.strides or contiguous_strides(source.shape),
        "position_stride": broadcast_strides(
            positions.shape, (source.shape[1],), positions.strides
        )[0],
        "position_dtype": native_dtype(positions.dtype),
        "interleaved": int(interleaved),
        "frequency_step": -2.0 * math.log(theta) / dim,
        "amplitude": float(amplitude),
    }


def avx2_rotary(context: KernelBindingContext) -> dict[str, ParameterValue]:
    result = rotary(context)
    strides = context.inputs["x"].strides or contiguous_strides(
        context.inputs["x"].shape
    )
    if strides[-1].op != "const" or strides[-1].value != 1:
        raise ValueError("AVX2 Rotary requires contiguous channels")
    return result


def attention(context: KernelBindingContext) -> dict[str, ParameterValue]:
    q, k, v = context.inputs["q"], context.inputs["k"], context.inputs["v"]
    if any(len(binding.shape) != 3 for binding in (q, k, v)):
        raise ValueError(
            "fused attention requires rank-three head/token/channel tensors"
        )
    if q.dtype != "fp32" or context.outputs[0].dtype != "fp32":
        raise ValueError("fused attention requires FP32 queries and output")
    causal = context.attrs.get("causal", True)
    scale = context.attrs.get("scale")
    window = context.attrs.get("window")
    if not isinstance(causal, bool) or (
        scale is not None and not isinstance(scale, (int, float))
    ):
        raise TypeError("invalid attention causal or scale attribute")
    if window is not None and (not isinstance(window, int) or window < 0):
        raise ValueError("fused attention requires a nonnegative window")
    q_rows, q_columns = storage_shape(q)
    k_rows, k_columns = storage_shape(k)
    v_rows, v_columns = storage_shape(v)
    return {
        "heads": q.shape[0],
        "kv_heads": k.shape[0],
        "queries": q.shape[1],
        "length": k.shape[1],
        "key_dim": q.shape[2],
        "value_dim": v.shape[2],
        "rank": 3,
        "q_strides": q.strides or contiguous_strides(q.shape),
        "k_strides": k.strides or contiguous_strides(k.shape),
        "v_strides": v.strides or contiguous_strides(v.shape),
        "q_rows": q_rows,
        "q_columns": q_columns,
        "k_rows": k_rows,
        "k_columns": k_columns,
        "v_rows": v_rows,
        "v_columns": v_columns,
        "window": window if isinstance(window, int) else 0,
        "q_dtype": DTYPE_IDS[q.dtype],
        "k_dtype": DTYPE_IDS[k.dtype],
        "v_dtype": DTYPE_IDS[v.dtype],
        "causal": int(causal),
        "windowed": int(window is not None),
        "default_scale": int(scale is None),
        "scale": float(scale) if isinstance(scale, (int, float)) else 1.0,
    }
