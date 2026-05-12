"""Quantization target parsing and naming."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuantizationTarget:
    value: str
    display_name: str


_TARGETS = {
    "auto": QuantizationTarget("auto", "auto"),
    "bf16": QuantizationTarget("bf16", "bf16"),
    "q8_0": QuantizationTarget("q8_0", "q8_0"),
    "q8_0_blocked_32": QuantizationTarget("q8_0_blocked_32", "q8_0_blocked_32"),
    "ternary": QuantizationTarget("ternary", "ternary"),
    "q1_0_128": QuantizationTarget("q1_0_128", "binary"),
    "grouped_ternary_128": QuantizationTarget("grouped_ternary_128", "grouped-ternary"),
}
_ALIASES = {
    "int8": "q8_0_blocked_32",
}

QUANTIZATION_CHOICES = tuple(sorted((*_TARGETS, *_ALIASES)))


def normalize_quantization(value: str | None) -> QuantizationTarget:
    raw_value = "auto" if value is None else str(value).strip().lower()
    canonical = _ALIASES.get(raw_value, raw_value)
    try:
        return _TARGETS[canonical]
    except KeyError as exc:
        choices = ", ".join(QUANTIZATION_CHOICES)
        raise ValueError(f"Unknown quantization {value!r}; expected one of: {choices}") from exc

