from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .tensor import Tensor
from .types import DType


class op_args(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )


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
    x: Tensor
    context: Tensor | None = None
    q_proj: str
    k_proj: str
    v_proj: str
    o_proj: str
    q_bias: str | None = None
    k_bias: str | None = None
    v_bias: str | None = None
    o_bias: str | None = None
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


class mla_args(op_args):
    x: Tensor
    context: Tensor | None = None
    q_proj: str | None = None
    q_a_proj: str | None = None
    q_a_norm: str | None = None
    q_b_proj: str | None = None
    kv_a_proj: str
    kv_a_norm: str
    kv_b_proj: str
    o_proj: str
    o_bias: str | None = None
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
    weight: str
    bias: str | None = None
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
    weight: str
    padding_idx: int | None = None


# FFN


class ffn_args(op_args):
    x: Tensor
    gate_proj: str | None = None
    up_proj: str
    down_proj: str
    gate_bias: str | None = None
    up_bias: str | None = None
    down_bias: str | None = None
    activation: ActivationOp


# Linear


class linear_args(op_args):
    x: Tensor
    weight: str
    bias: str | None = None
    transpose_weight: bool = False


# MoE


class moe_args(op_args):
    x: Tensor
    router_weight: str
    router_bias: str | None = None
    num_experts: int = Field(gt=0)
    experts_per_token: int = Field(gt=0)
    expert_type: ActivationOp
    gate_weights: list[str] | str | None = None
    up_weights: list[str] | str
    down_weights: list[str] | str
    gate_biases: list[str] | str | None = None
    up_biases: list[str] | str | None = None
    down_biases: list[str] | str | None = None
    router_scoring: Literal["softmax", "sigmoid"] = "softmax"
    normalize_weights: bool = False
    shared_experts: int = Field(default=0, ge=0)
    shared_gate_weight: str | None = None

    @model_validator(mode="after")
    def validate_expert_count(self) -> moe_args:
        if self.experts_per_token > self.num_experts:
            raise ValueError("experts_per_token cannot exceed num_experts")

        return self


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
    weight: str | None = None
    bias: str | None = None
    eps: float = Field(default=1e-5, gt=0)


class rmsnorm_args(op_args):
    x: Tensor
    weight: str
    eps: float = Field(default=1e-6, gt=0)


class batchnorm_args(op_args):
    x: Tensor
    weight: str | None = None
    bias: str | None = None
    running_mean: str
    running_var: str
    eps: float = Field(default=1e-5, gt=0)


class groupnorm_args(op_args):
    x: Tensor
    weight: str | None = None
    bias: str | None = None
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
    in_proj: str
    conv_weight: str
    conv_bias: str | None = None
    x_proj: str
    dt_proj: str
    dt_bias: str | None = None
    a_log: str
    d: str
    out_proj: str
    out_bias: str | None = None
    state_dim: int = Field(gt=0)
    conv_kernel: int = Field(gt=0)


class mamba2_args(op_args):
    x: Tensor
    in_proj: str
    conv_weight: str
    conv_bias: str | None = None
    a_log: str
    d: str
    dt_bias: str | None = None
    norm_weight: str | None = None
    out_proj: str
    out_bias: str | None = None
    state_dim: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    num_groups: int = Field(gt=0)
    conv_kernel: int = Field(gt=0)
