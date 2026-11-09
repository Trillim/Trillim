from __future__ import annotations

from typing import get_args

from ...tdl.core.tensor import Tensor
from ...tdl.core.types import DType
from ..compiler_dataclasses import _ShapeExpression
from ..passes.infer import _constant_expression, _expression
from .kernel_dataclasses import _KernelBindingContext, _ParameterValue, _TensorBinding

_DTYPE_IDS: dict[DType, int] = {
    name: index for index, name in enumerate(get_args(DType))
}


def _product(shape: tuple[_ShapeExpression, ...]) -> _ShapeExpression:
    result = _constant_expression(1)
    for dimension in shape:
        result = _expression("mul", result, dimension)
    return result


def _contiguous_strides(
    shape: tuple[_ShapeExpression, ...],
) -> tuple[_ShapeExpression, ...]:
    strides: list[_ShapeExpression] = []
    size = _constant_expression(1)
    for dimension in reversed(shape):
        strides.append(size)
        size = _expression("mul", size, dimension)
    return tuple(reversed(strides))


def _broadcast_strides(
    shape: tuple[_ShapeExpression, ...],
    target: tuple[_ShapeExpression, ...],
    strides: tuple[_ShapeExpression, ...] | None = None,
) -> tuple[_ShapeExpression, ...]:
    if len(shape) > len(target):
        raise ValueError("cannot broadcast to a smaller rank")
    strides = strides or _contiguous_strides(shape)
    result = [_constant_expression(0)] * (len(target) - len(shape))
    for dimension, stride, output in zip(
        shape, strides, target[len(target) - len(shape) :], strict=True
    ):
        if dimension.op == "const" and dimension.value == 1:
            result.append(_constant_expression(0))
        elif dimension == output:
            result.append(stride)
        else:
            result.append(_expression("broadcast_stride", dimension, stride))
    return tuple(result)


def _native_dtype(dtype: DType) -> int:
    result = _DTYPE_IDS[dtype]
    if result > _DTYPE_IDS["fp32"]:
        raise ValueError("kernel requires native tensor storage")
    return result


def _binary(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    output = context.outputs[0]
    a, b = context.inputs["a"], context.inputs["b"]
    if a.dtype != output.dtype or b.dtype != output.dtype:
        raise ValueError("binary inputs must have the promoted output dtype")
    return {
        "count": _product(output.shape),
        "rank": len(output.shape),
        "shape": output.shape,
        "a_strides": _broadcast_strides(a.shape, output.shape, a.strides),
        "b_strides": _broadcast_strides(b.shape, output.shape, b.strides),
        "dtype": _native_dtype(output.dtype),
    }


def _unary(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    output = context.outputs[0]
    source = context.inputs["x"]
    if source.dtype != output.dtype:
        raise ValueError("unary input must have the output's physical dtype")
    return {
        "count": _product(output.shape),
        "rank": len(output.shape),
        "shape": output.shape,
        "strides": source.strides or _contiguous_strides(source.shape),
        "dtype": _native_dtype(output.dtype),
    }


def _storage_shape(
    binding: _TensorBinding,
) -> tuple[_ShapeExpression, _ShapeExpression]:
    if binding.storage_shape is not None:
        return binding.storage_shape
    if len(binding.shape) < 2:
        return _constant_expression(1), _product(binding.shape)
    return binding.shape[0], _product(binding.shape[1:])


def _batch_strides(
    binding: _TensorBinding, target: tuple[_ShapeExpression, ...]
) -> tuple[_ShapeExpression, ...]:
    shape = binding.shape[:-2]
    strides = (binding.strides or _contiguous_strides(binding.shape))[:-2]
    result = [_constant_expression(0)] * (len(target) - len(shape))
    for dimension, stride, output in zip(
        shape, strides, target[len(target) - len(shape) :], strict=True
    ):
        if dimension.op == "const" and dimension.value == 1:
            result.append(_constant_expression(0))
        elif dimension == output:
            result.append(stride)
        else:
            result.append(_expression("broadcast_stride", dimension, stride))
    return tuple(result)


def _matmul(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    a, b = context.inputs["a"], context.inputs["b"]
    output = context.outputs[0]
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError("matrix multiplication requires rank at least two")
    if (not output.dtype.startswith("fp") and output.dtype != "bf16") and (
        _DTYPE_IDS[a.dtype] >= _DTYPE_IDS["fp16"]
        or _DTYPE_IDS[b.dtype] >= _DTYPE_IDS["fp16"]
    ):
        raise ValueError("floating matrix inputs require a floating output")
    batch = output.shape[:-2]
    a_strides = a.strides or _contiguous_strides(a.shape)
    b_strides = b.strides or _contiguous_strides(b.shape)
    a_rows, a_columns = _storage_shape(a)
    b_rows, b_columns = _storage_shape(b)
    return {
        "batches": _product(batch),
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
        "a_dtype": _DTYPE_IDS[a.dtype],
        "b_dtype": _DTYPE_IDS[b.dtype],
        "out_dtype": _native_dtype(output.dtype),
        "a_storage_rows": a_rows,
        "a_storage_columns": a_columns,
        "b_storage_rows": b_rows,
        "b_storage_columns": b_columns,
    }


def _cast(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    source = context.inputs["x"]
    output = context.outputs[0]
    source_rows, source_columns = _storage_shape(source)
    output_rows, output_columns = _storage_shape(output)
    return {
        "count": _product(output.shape),
        "rank": len(source.shape),
        "shape": source.shape,
        "strides": source.strides or _contiguous_strides(source.shape),
        "source_offset": 0,
        "source_rows": source_rows,
        "source_columns": source_columns,
        "output_rows": output_rows,
        "output_columns": output_columns,
        "input_dtype": _DTYPE_IDS[source.dtype],
        "output_dtype": _DTYPE_IDS[output.dtype],
    }


def _reduction(
    context: _KernelBindingContext, axes: tuple[int, ...]
) -> dict[str, _ParameterValue]:
    source = context.inputs["x"]
    rank = len(source.shape)
    normalized = tuple(axis % rank for axis in axes)
    strides = source.strides or _contiguous_strides(source.shape)
    kept = tuple(axis for axis in range(rank) if axis not in normalized)
    rows, columns = _storage_shape(source)
    return {
        "outputs": _product(context.outputs[0].shape),
        "output_rank": len(kept),
        "reduce_rank": len(normalized),
        "reduce_count": _product(tuple(source.shape[axis] for axis in normalized)),
        "output_shape": tuple(source.shape[axis] for axis in kept),
        "output_strides": tuple(strides[axis] for axis in kept),
        "reduce_shape": tuple(source.shape[axis] for axis in normalized),
        "reduce_strides": tuple(strides[axis] for axis in normalized),
        "storage_rows": rows,
        "storage_columns": columns,
        "input_dtype": _DTYPE_IDS[source.dtype],
        "output_dtype": _native_dtype(context.outputs[0].dtype),
    }


def _bind_reduction(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    axes = context.attrs["axes"]
    if not isinstance(axes, tuple) or not all(isinstance(axis, int) for axis in axes):
        raise TypeError("reduction axes must be an integer tuple")
    return _reduction(context, axes)


def _argmax(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    axis = context.attrs.get("axis", -1)
    if not isinstance(axis, int):
        raise TypeError("argmax axis must be an integer")
    return _reduction(context, (axis,))


def _compare(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    output = context.outputs[0]
    if output.dtype != "bool":
        raise ValueError("comparison output must have bool storage")
    dtype = context.inputs["a"].dtype
    promoted = _TensorBinding(output.shape, output.maximum_shape, dtype)
    result = _binary(_KernelBindingContext(context.inputs, (promoted,), context.attrs))
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


def _select(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    result = _binary(
        _KernelBindingContext(
            {"a": context.inputs["when_true"], "b": context.inputs["when_false"]},
            context.outputs,
            context.attrs,
        )
    )
    condition = context.inputs["condition"]
    if condition.dtype != "bool":
        raise ValueError("selection condition must have bool storage")
    result["condition_strides"] = _broadcast_strides(
        condition.shape, context.outputs[0].shape, condition.strides
    )
    return result


def _permute(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    result = _cast(context)
    axes = context.attrs["axes"]
    if not isinstance(axes, tuple) or not all(isinstance(axis, int) for axis in axes):
        raise TypeError("permute axes must be an integer tuple")
    source = context.inputs["x"]
    strides = source.strides or _contiguous_strides(source.shape)
    result.update(
        {
            "rank": len(context.outputs[0].shape),
            "shape": context.outputs[0].shape,
            "strides": tuple(strides[axis] for axis in axes),
        }
    )
    return result


def _broadcast(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    result = _cast(context)
    source = context.inputs["x"]
    output = context.outputs[0]
    result.update(
        {
            "rank": len(output.shape),
            "shape": output.shape,
            "strides": _broadcast_strides(source.shape, output.shape, source.strides),
        }
    )
    return result


def _shape(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    if context.outputs[0].dtype != "int64":
        raise ValueError("shape output must have int64 storage")
    source = context.inputs["x"]
    return {"rank": len(source.shape), "shape": source.shape}


def _scalar_parameter(
    value: object, binding: _TensorBinding | None = None
) -> int | _ShapeExpression:
    if isinstance(value, (int, _ShapeExpression)):
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


def _iota(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    return {
        "count": _product(context.outputs[0].shape),
        "start": _scalar_parameter(
            context.attrs.get("start", 0), context.inputs.get("start")
        ),
        "step": _scalar_parameter(context.attrs.get("step", 1)),
        "dtype": _native_dtype(context.outputs[0].dtype),
    }


def _random(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    distribution = context.attrs.get("distribution", "uniform")
    if distribution not in ("uniform", "normal"):
        raise ValueError("invalid random distribution")
    return {
        "count": _product(context.outputs[0].shape),
        "seed": _scalar_parameter(context.attrs["seed"], context.inputs.get("seed")),
        "offset": _scalar_parameter(
            context.attrs.get("offset", 0), context.inputs.get("offset")
        ),
        "distribution": int(distribution == "normal"),
        "dtype": _native_dtype(context.outputs[0].dtype),
    }


def _indexing(
    context: _KernelBindingContext, scatter_operation: bool
) -> dict[str, _ParameterValue]:
    source = context.inputs["x"]
    indices = context.inputs["indices"]
    output = context.outputs[0]
    if (
        source.dtype != output.dtype
        or scatter_operation
        and context.inputs["values"].dtype != output.dtype
    ):
        raise ValueError("indexing data must have the output's physical dtype")
    coordinates = indices.shape[-1].value
    if coordinates is None or coordinates < 0 or coordinates > len(source.shape):
        raise ValueError("indexed coordinate width must be statically known")
    if indices.dtype != "int64":
        raise ValueError("index kernels require promoted int64 coordinates")
    source_strides = source.strides or _contiguous_strides(source.shape)
    index_strides = indices.strides or _contiguous_strides(indices.shape)
    index_shape = indices.shape[:-1]
    slice_shape = source.shape[coordinates:]
    target_shape = (*index_shape, *slice_shape)
    storage_rows, storage_columns = _storage_shape(source)
    output_rows, output_columns = _storage_shape(output)
    mask = context.inputs.get("mask")
    if mask is not None and mask.dtype != "bool":
        raise ValueError("indexing masks must have bool storage")
    fill = context.attrs.get("fill", 0)
    if not isinstance(fill, (int, float)):
        raise TypeError("gather fill must be a numeric scalar")
    mode = context.attrs.get("mode", "replace")
    if mode not in ("replace", "add"):
        raise ValueError("invalid scatter mode")
    if scatter_operation:
        _native_dtype(output.dtype)
    floating = (
        output.dtype in ("fp16", "bf16", "fp32")
        or _DTYPE_IDS[output.dtype] >= _DTYPE_IDS["row_int8"]
    )
    values = context.inputs.get("values")
    return {
        "count": _product(output.shape),
        "items": _product(index_shape),
        "coordinates": coordinates,
        "index_rank": len(index_shape),
        "slice_rank": len(slice_shape),
        "slice_size": _product(slice_shape),
        "index_shape": index_shape,
        "index_strides": index_strides[:-1],
        "mask_strides": _broadcast_strides(mask.shape, index_shape, mask.strides)
        if mask
        else (_constant_expression(0),) * len(index_shape),
        "coordinate_stride": index_strides[-1],
        "indexed_shape": source.shape[:coordinates],
        "indexed_strides": source_strides[:coordinates],
        "output_indexed_strides": _contiguous_strides(source.shape)[:coordinates],
        "slice_shape": slice_shape,
        "slice_strides": source_strides[coordinates:],
        "values_rank": len(target_shape) if values else 0,
        "values_shape": target_shape if values else (),
        "values_strides": _broadcast_strides(values.shape, target_shape, values.strides)
        if values
        else (),
        "storage_rows": storage_rows,
        "storage_columns": storage_columns,
        "output_rows": output_rows,
        "output_columns": output_columns,
        "dtype": _DTYPE_IDS[output.dtype],
        "masked": int(mask is not None),
        "mode": int(mode == "add"),
        "fill_bits": 0
        if floating
        else ((int(bool(fill)) if output.dtype == "bool" else int(fill)) & (2**64 - 1)),
        "fill_float": float(fill),
    }


def _gather(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    return _indexing(context, False)


def _scatter(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    return _indexing(context, True)


def _reshape_strides(
    shape: tuple[_ShapeExpression, ...],
    strides: tuple[_ShapeExpression, ...],
    target: tuple[_ShapeExpression, ...],
) -> tuple[_ShapeExpression, ...] | None:
    if len(shape) != len(strides):
        raise ValueError("tensor shape and strides must have equal rank")
    expected = _constant_expression(1)
    contiguous = True
    for dimension, stride in zip(reversed(shape), reversed(strides), strict=True):
        if dimension.op != "const" or dimension.value != 1:
            contiguous = contiguous and stride == expected
        expected = _expression("mul", expected, dimension)
    if contiguous:
        return _contiguous_strides(target)
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
        _constant_expression(0)
        if dimension.op == "const" and dimension.value == 1
        else next(iterator)[1]
        for dimension in target
    )


def _rmsnorm(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    source = context.inputs["x"]
    if not source.shape:
        raise ValueError("RMSNorm requires a channel axis")
    source_strides = source.strides or _contiguous_strides(source.shape)
    if _reshape_strides(
        source.shape, source_strides, source.shape
    ) != _contiguous_strides(source.shape):
        raise ValueError("fused RMSNorm requires contiguous input")
    rows, columns = _storage_shape(source)
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
        "rows": _product(source.shape[:-1]),
        "width": source.shape[-1],
        "storage_rows": rows,
        "storage_columns": columns,
        "input_dtype": _DTYPE_IDS[source.dtype],
        "weight_dtype": _DTYPE_IDS[weight.dtype] if weight else _DTYPE_IDS["fp32"],
        "output_dtype": _native_dtype(context.outputs[0].dtype),
        "eps": float(eps),
        "weight_offset": float(offset),
    }


def _linear(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    source = context.inputs["x"]
    weight = context.inputs["weight"]
    output = context.outputs[0]
    if not source.shape or len(weight.shape) != 2:
        raise ValueError("linear requires a channel axis and matrix weights")
    rows = _product(source.shape[:-1])
    source_shape = (rows, source.shape[-1])
    source_strides = _reshape_strides(
        source.shape, source.strides or _contiguous_strides(source.shape), source_shape
    )
    if source_strides is None:
        raise ValueError("fused linear requires a matrix-representable input view")
    matrix = _TensorBinding(
        source_shape,
        (rows.maximum, source.maximum_shape[-1]),
        source.dtype,
        source_strides,
        _storage_shape(source),
    )
    strides = weight.strides or _contiguous_strides(weight.shape)
    transposed = _TensorBinding(
        (weight.shape[1], weight.shape[0]),
        (weight.maximum_shape[1], weight.maximum_shape[0]),
        weight.dtype,
        (strides[1], strides[0]),
        _storage_shape(weight),
    )
    result = _TensorBinding(
        (rows, weight.shape[0]), (rows.maximum, weight.maximum_shape[0]), output.dtype
    )
    return _matmul(
        _KernelBindingContext({"a": matrix, "b": transposed}, (result,), context.attrs)
    )


def _bitlinear(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
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
        strides = binding.strides or _contiguous_strides(binding.shape)
        if _reshape_strides(
            binding.shape, strides, binding.shape
        ) != _contiguous_strides(binding.shape):
            raise ValueError("fused BitLinear requires contiguous operands")
    return {
        "tokens": _product(source.shape[:-1]),
        "width": source.shape[-1],
        "outputs": weight.shape[0],
    }


def _quantize_int8(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    source = context.inputs["x"]
    if not source.shape or source.dtype not in {"fp16", "bf16", "fp32"}:
        raise ValueError("INT8 activation quantization requires floating rows")
    if (source.strides or _contiguous_strides(source.shape)) != _contiguous_strides(
        source.shape
    ):
        raise ValueError("INT8 activation quantization requires contiguous input")
    codes, scale = context.outputs
    if (
        codes.dtype != "int8"
        or codes.shape != source.shape
        or scale.dtype != "fp32"
        or scale.shape != (*source.shape[:-1], _constant_expression(1))
    ):
        raise ValueError("INT8 activation codes or scales have incompatible metadata")
    return {
        "tokens": _product(source.shape[:-1]),
        "width": source.shape[-1],
        "input_dtype": _DTYPE_IDS[source.dtype],
    }


def _quantized_linear(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    source, weight, scale = (context.inputs[name] for name in ("x", "weight", "scale"))
    if (
        not source.shape
        or len(weight.shape) != 2
        or source.shape[-1] != weight.shape[-1]
        or source.dtype != "int8"
        or weight.dtype != "mat_ternary"
        or scale.dtype != "fp32"
        or scale.shape != (*source.shape[:-1], _constant_expression(1))
        or context.outputs[0].dtype not in {"fp16", "bf16", "fp32"}
    ):
        raise ValueError(
            "quantized linear requires INT8 rows, ternary weights and FP32 scales"
        )
    for binding in (source, weight, scale):
        if (
            binding.strides or _contiguous_strides(binding.shape)
        ) != _contiguous_strides(binding.shape):
            raise ValueError("quantized linear requires contiguous operands")
    return {
        "tokens": _product(source.shape[:-1]),
        "width": source.shape[-1],
        "outputs": weight.shape[0],
        "output_dtype": _DTYPE_IDS[context.outputs[0].dtype],
    }


def _avx2_bitlinear(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    result = _bitlinear(context)
    width = context.inputs["x"].shape[-1]
    if width.op != "const" or width.value is None or width.value % 128:
        raise ValueError("AVX2 BitLinear requires a fixed width divisible by 128")
    return result


def _avx2_linear(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    result = _linear(context)
    if context.inputs["x"].dtype != "fp32" or context.outputs[0].dtype != "fp32":
        raise ValueError("AVX2 linear requires FP32 activations and output")
    for name in ("a_col_stride", "b_row_stride"):
        stride = result[name]
        if (
            not isinstance(stride, _ShapeExpression)
            or stride.op != "const"
            or stride.value != 1
        ):
            raise ValueError("AVX2 linear requires contiguous contraction axes")
    return result


def _avx2_rmsnorm(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    result = _rmsnorm(context)
    if context.inputs["x"].dtype != "fp32" or context.outputs[0].dtype != "fp32":
        raise ValueError("AVX2 RMSNorm requires FP32 activations and output")
    return result


def _activation(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    output = context.outputs[0]
    for name in ("a", "b"):
        source = context.inputs[name]
        strides = source.strides or _contiguous_strides(source.shape)
        if (
            source.dtype != "fp32"
            or source.shape != output.shape
            or strides != _contiguous_strides(source.shape)
        ):
            raise ValueError(
                "fused gate activation requires matching contiguous FP32 operands"
            )
    if output.dtype != "fp32":
        raise ValueError("fused gate activation requires FP32 output")
    return {"count": _product(output.shape)}


def _rotary(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    import math

    source = context.inputs["x"]
    positions = context.inputs["positions"]
    if (
        len(source.shape) != 3
        or len(positions.shape) != 1
        or source.dtype not in {"fp16", "bf16", "fp32"}
        or context.outputs[0].dtype not in {"fp16", "bf16", "fp32"}
    ):
        raise ValueError(
            "fused Rotary requires floating [tokens,heads,channels] and one-dimensional positions"
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
        "heads": source.shape[1],
        "tokens": source.shape[0],
        "dim": source.shape[2],
        "rotary_dim": dim,
        "rank": 3,
        "strides": tuple(
            (source.strides or _contiguous_strides(source.shape))[axis]
            for axis in (1, 0, 2)
        ),
        "input_dtype": _DTYPE_IDS[source.dtype],
        "output_dtype": _DTYPE_IDS[context.outputs[0].dtype],
        "position_stride": _broadcast_strides(
            positions.shape, (source.shape[0],), positions.strides
        )[0],
        "position_dtype": _native_dtype(positions.dtype),
        "interleaved": int(interleaved),
        "frequency_step": -2.0 * math.log(theta) / dim,
        "amplitude": float(amplitude),
    }


def _avx2_rotary(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    result = _rotary(context)
    strides = context.inputs["x"].strides or _contiguous_strides(
        context.inputs["x"].shape
    )
    if strides[-1].op != "const" or strides[-1].value != 1:
        raise ValueError("AVX2 Rotary requires contiguous channels")
    return result


def _attention(context: _KernelBindingContext) -> dict[str, _ParameterValue]:
    q, k, v = context.inputs["q"], context.inputs["k"], context.inputs["v"]
    if any(len(binding.shape) != 3 for binding in (q, k, v)):
        raise ValueError(
            "fused attention requires rank-three token/head/channel tensors"
        )
    if q.dtype not in {"fp16", "bf16", "fp32"} or context.outputs[0].dtype not in {
        "fp16",
        "bf16",
        "fp32",
    }:
        raise ValueError("fused attention requires floating queries and output")
    causal = context.attrs.get("causal", True)
    scale = context.attrs.get("scale")
    window = context.attrs.get("window")
    if not isinstance(causal, bool) or (
        scale is not None and not isinstance(scale, (int, float))
    ):
        raise TypeError("invalid attention causal or scale attribute")
    if window is not None and (not isinstance(window, int) or window < 0):
        raise ValueError("fused attention requires a nonnegative window")
    q_rows, q_columns = _storage_shape(q)
    k_rows, k_columns = _storage_shape(k)
    v_rows, v_columns = _storage_shape(v)
    return {
        "heads": q.shape[1],
        "kv_heads": k.shape[1],
        "queries": q.shape[0],
        "length": k.shape[0],
        "key_dim": q.shape[2],
        "value_dim": v.shape[2],
        "rank": 3,
        "q_strides": tuple(
            (q.strides or _contiguous_strides(q.shape))[axis] for axis in (1, 0, 2)
        ),
        "k_strides": tuple(
            (k.strides or _contiguous_strides(k.shape))[axis] for axis in (1, 0, 2)
        ),
        "v_strides": tuple(
            (v.strides or _contiguous_strides(v.shape))[axis] for axis in (1, 0, 2)
        ),
        "q_rows": q_rows,
        "q_columns": q_columns,
        "k_rows": k_rows,
        "k_columns": k_columns,
        "v_rows": v_rows,
        "v_columns": v_columns,
        "window": window if isinstance(window, int) else 0,
        "output_dtype": _DTYPE_IDS[context.outputs[0].dtype],
        "q_dtype": _DTYPE_IDS[q.dtype],
        "k_dtype": _DTYPE_IDS[k.dtype],
        "v_dtype": _DTYPE_IDS[v.dtype],
        "causal": int(causal),
        "windowed": int(window is not None),
        "default_scale": int(scale is None),
        "scale": float(scale) if isinstance(scale, (int, float)) else 1.0,
    }
