from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .tdl_dataclasses import Shape
from .tensor import Tensor
from .types import DType


class Weight(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    root: str
    shape: Shape | None = None
    dtype: DType | None = None

    def __init__(self, root: str, *, dtype: DType | None = None) -> None:
        super().__init__(root=root, dtype=dtype)

    @field_validator("root")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        model_name, separator, tensor_name = value.partition(".")

        if not model_name.isalpha():
            raise ValueError("model names must contain only letters")

        if not separator or not tensor_name:
            raise ValueError("weight must name one exact tensor as Model.tensor_name")

        return value

    @property
    def model_name(self) -> str:
        return self.root.partition(".")[0]

    @property
    def tensor_name(self) -> str:
        return self.root.partition(".")[2]

    def __str__(self) -> str:
        return self.root


class op_args(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )


class LoRA(op_args):
    """An ordered (down/A, up/B) pair; absent alpha means unit scale."""

    down_up: tuple[Weight, Weight]
    alpha: float | None = None


class Projection(op_args):
    weight: Weight
    bias: Weight | None = None
    loras: tuple[LoRA, ...] = ()


# Activation


class relu_args(op_args):
    x: Tensor


class sigmoid_args(op_args):
    x: Tensor


class tanh_args(op_args):
    x: Tensor


class gelu_args(op_args):
    x: Tensor
    approximate: Literal["none", "tanh"] = "none"


class silu_args(op_args):
    x: Tensor


class relu_sqr_args(op_args):
    x: Tensor


# These models must exist before activation.py imports this module.
from .activation import ActivationOp  # noqa: I001


# Attention


class gqa_args(op_args):
    """Return concatenated attention heads before the output projection."""

    x: Tensor
    context: Tensor | None = None
    q_proj: Projection
    k_proj: Projection
    v_proj: Projection
    num_heads: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    causal: bool = True
    scale: float | None = None
    window_size: int | None = Field(default=None, gt=0)
    position: Literal["none", "rope"] = "none"
    rope_theta: float | None = Field(default=None, gt=0)
    rope_dim: int | None = Field(default=None, gt=0)
    yarn_factor: float | None = Field(default=None, gt=0)
    yarn_original_max_position_embeddings: int | None = Field(default=None, gt=0)
    yarn_beta_fast: float | None = None
    yarn_beta_slow: float | None = None
    yarn_mscale: float | None = None
    yarn_mscale_all_dim: float | None = None
    yarn_attention_factor: float | None = None


class gqa_norm_args(gqa_args):
    """Apply optional per-head RMS norms to Q, K, and V after projection.

    Q and K norms run before positional encoding. A weight offset of one
    supports checkpoints that store zero-centered RMS norm weights.
    """

    q_norm: Weight | None = None
    k_norm: Weight | None = None
    v_norm: Weight | None = None
    norm_eps: float = Field(default=1e-6, gt=0)
    norm_weight_offset: Literal[0, 1] = 0


class gated_delta_args(op_args):
    """Apply causal delta-rule attention and gated RMS norm before output Linear.

    The QKV projection packs query, key, and value in that order. The
    convolution and recurrent states are implicit.
    """

    x: Tensor
    qkv_proj: Projection
    z_proj: Projection
    b_proj: Projection
    a_proj: Projection
    conv_weight: Weight
    conv_bias: Weight | None = None
    dt_bias: Weight
    a_log: Weight
    norm_weight: Weight
    num_key_heads: int = Field(gt=0)
    conv_activation: ActivationOp = "silu"
    norm_eps: float = Field(default=1e-6, gt=0)


class gated_full_attn_args(gqa_norm_args):
    """Q projection packs (query, gate) pairs per head.

    Apply sigmoid to each gate and multiply it with the attention context
    before the separate output Linear operation.
    """


class mla_args(op_args):
    """Return concatenated attention heads before the output projection."""

    x: Tensor
    context: Tensor | None = None
    q_proj: Projection | None = None
    q_a_proj: Projection | None = None
    q_a_norm: Weight | None = None
    q_b_proj: Projection | None = None
    kv_a_proj: Projection
    kv_a_norm: Weight
    kv_b_proj: Projection
    num_heads: int = Field(gt=0)
    q_lora_rank: int | None = Field(default=None, gt=0)
    kv_lora_rank: int = Field(gt=0)
    qk_nope_head_dim: int = Field(gt=0)
    qk_rope_head_dim: int = Field(gt=0)
    v_head_dim: int = Field(gt=0)
    causal: bool = False
    scale: float | None = None
    rope_theta: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_query_projection(self) -> mla_args:
        direct = self.q_proj is not None
        low_rank = self.q_a_proj is not None and self.q_b_proj is not None

        if direct == low_rank or (self.q_a_proj is None) != (self.q_b_proj is None):
            raise ValueError("provide either q_proj or both q_a_proj and q_b_proj")

        return self


# Cast


class cast_args(op_args):
    x: Tensor
    dtype: DType


# Conv


class conv_args(op_args):
    x: Tensor
    weight: Weight
    bias: Weight | None = None
    dimensions: Literal[1, 2, 3]
    transpose: bool = False
    stride: int | tuple[int, ...] = 1
    padding: int | tuple[int, ...] | Literal["same", "valid"] = 0
    dilation: int | tuple[int, ...] = 1
    groups: int = Field(default=1, gt=0)
    output_padding: int | tuple[int, ...] = 0
    resize_mode: (
        Literal["nearest", "linear", "bilinear", "bicubic", "trilinear"] | None
    ) = None
    resize_scale: float | tuple[float, ...] | None = None


# Embedding


class embedding_args(op_args):
    x: Tensor
    weight: Weight
    padding_idx: int | None = None


# FFN


class ffn_args(op_args):
    x: Tensor
    gate_proj: Projection | None = None
    up_proj: Projection
    down_proj: Projection
    activation: ActivationOp


# Linear


class linear_args(op_args):
    x: Tensor
    projection: Projection
    transpose_weight: bool = False


# MoE


class ExpertProjections(op_args):
    """One expert's projections; gate is optional for ungated experts."""

    gate: Projection | None = None
    up: Projection
    down: Projection


class moe_args(op_args):
    """The router's output index selects the expert at the same list index.

    SparseMixer selects two experts in sequence. At each step it masks scores
    using router_jitter_eps, then takes the selected expert's softmax weight
    over the unmasked candidates. The two weights are not jointly normalized.
    """

    x: Tensor
    router: Projection
    experts: list[ExpertProjections] = Field(min_length=1)
    experts_per_token: int = Field(gt=0)
    expert_type: ActivationOp
    router_scoring: Literal["softmax", "sigmoid", "sparsemixer"] = "softmax"
    router_jitter_eps: float | None = Field(default=None, ge=0)
    normalize_weights: bool = False

    @model_validator(mode="after")
    def validate_expert_count(self) -> moe_args:
        if self.experts_per_token > len(self.experts):
            raise ValueError("experts_per_token cannot exceed the number of experts")

        if self.router_scoring == "sparsemixer":
            if self.experts_per_token != 2:
                raise ValueError("SparseMixer requires exactly two experts per token")
            if self.router_jitter_eps is None:
                raise ValueError("SparseMixer requires router_jitter_eps")
        elif self.router_jitter_eps is not None:
            raise ValueError("router_jitter_eps is only used with SparseMixer")

        return self


class shared_moe_args(moe_args):
    """Combine routed experts with always-active shared experts."""

    shared_experts: list[ExpertProjections] = Field(min_length=1)


# Math


class add_args(op_args):
    a: Tensor
    b: Tensor | float | int


class mul_args(op_args):
    a: Tensor
    b: Tensor | float | int


class concat_args(op_args):
    tensors: list[Tensor] = Field(min_length=1)
    axis: int


class softmax_args(op_args):
    x: Tensor
    axis: int = -1


# Norm


class layernorm_args(op_args):
    x: Tensor
    weight: Weight | None = None
    bias: Weight | None = None
    eps: float = Field(default=1e-5, gt=0)


class rmsnorm_args(op_args):
    x: Tensor
    weight: Weight
    eps: float = Field(default=1e-6, gt=0)


class batchnorm_args(op_args):
    x: Tensor
    weight: Weight | None = None
    bias: Weight | None = None
    running_mean: Weight
    running_var: Weight
    eps: float = Field(default=1e-5, gt=0)


class groupnorm_args(op_args):
    x: Tensor
    weight: Weight | None = None
    bias: Weight | None = None
    num_groups: int = Field(gt=0)
    eps: float = Field(default=1e-5, gt=0)


# Pool


class max_pool_args(op_args):
    x: Tensor
    dimensions: Literal[1, 2, 3]
    kernel_size: int | tuple[int, ...]
    stride: int | tuple[int, ...] | None = None
    padding: int | tuple[int, ...] = 0
    dilation: int | tuple[int, ...] = 1
    ceil_mode: bool = False


class avg_pool_args(op_args):
    x: Tensor
    dimensions: Literal[1, 2, 3]
    kernel_size: int | tuple[int, ...]
    stride: int | tuple[int, ...] | None = None
    padding: int | tuple[int, ...] = 0
    ceil_mode: bool = False
    count_include_pad: bool = True


class adaptive_avg_pool_args(op_args):
    x: Tensor
    output_size: int | tuple[int, ...]


class adaptive_max_pool_args(op_args):
    x: Tensor
    output_size: int | tuple[int, ...]


# SSM


class mamba_args(op_args):
    x: Tensor
    in_proj: Weight
    conv_weight: Weight
    conv_bias: Weight | None = None
    x_proj: Weight
    dt_proj: Weight
    dt_bias: Weight | None = None
    a_log: Weight
    d: Weight
    out_proj: Weight
    out_bias: Weight | None = None
    state_dim: int = Field(gt=0)
    conv_kernel: int = Field(gt=0)


class mamba2_args(op_args):
    x: Tensor
    in_proj: Weight
    conv_weight: Weight
    conv_bias: Weight | None = None
    a_log: Weight
    d: Weight
    dt_bias: Weight | None = None
    norm_weight: Weight | None = None
    out_proj: Weight
    out_bias: Weight | None = None
    state_dim: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    num_groups: int = Field(gt=0)
    conv_kernel: int = Field(gt=0)
