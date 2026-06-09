"""Image-model quantization policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ImageTensorAction(IntEnum):
    """Numeric tensor storage actions expected by the image quantizer/runtime."""

    BF16_RAW = 0
    Q1_0_128 = 4
    GROUP_TERNARY = 5
    Q4_0 = 6


@dataclass(frozen=True, slots=True)
class ImageQuantization:
    name: str
    transformer_weight_action: ImageTensorAction
    text_encoder_weight_action: ImageTensorAction
