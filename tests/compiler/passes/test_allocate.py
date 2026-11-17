from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.compiler.compiler_dataclasses import PlannedCase
from tests.compiler.model_cases import CASE_NAMES
from trillim.compiler.artifacts.artifact_dataclasses import _Buffer, _Workspace

# DarkNet old-main: include/model/{model.h,scratch_space.h} reuses residual,
# gate/up and attention banks. Budgets below scale that manual plan to T=4,D=32.
_HAND_BYTES = {
    "projection": 512,  # one projection output
    "views": 0,  # head splitting is metadata only
    "softmax": 512,  # output; native workspace is added separately
    "swiglu": 1536,  # normalized input, gate and up; reuse gate for product
    "gqa": 1536,  # Q, K/V and output; reuse dead projection storage
    "kv_cache": 1664,  # GQA plus a retained two-token key snapshot
    "recurrence": 704,  # three output rows, two state banks and loop scalar
    "conv_branch_reduce": 192,  # convolution/branch output and two moments
}


@pytest.mark.parametrize("name", CASE_NAMES)
def test_allocation_evaluation(
    name: str,
    model_plans: Callable[[str], PlannedCase],
    record_property: Callable[[str, object], None],
) -> None:
    planned = model_plans(name)
    layout = planned.layout
    ideal = _HAND_BYTES[name]
    workspace = max((span.size for span in layout.scratch.values()), default=0)
    print(
        f"{name}: arena={layout.arena_size}, hand_activation_budget={ideal}, kernel_workspace_peak={workspace}"
    )
    record_property("arena_bytes", layout.arena_size)
    record_property("hand_activation_bytes", ideal)
    record_property("kernel_workspace_peak", workspace)
    record_property(
        "over_hand_plan",
        layout.arena_size / (ideal + workspace) if ideal + workspace else 1.0,
    )
    assert layout.arena_size >= 0
    for buffer in layout.buffers.values():
        if buffer.storage != "arena" or buffer.first > buffer.last:
            continue
        assert buffer.offset % buffer.alignment == 0
        assert buffer.offset + buffer.size <= layout.arena_size
    arena = [
        b for b in layout.buffers.values() if b.storage == "arena" and b.first <= b.last
    ]
    spans: list[_Buffer | _Workspace] = [
        *arena,
        *layout.scratch.values(),
        *layout.transfer_scratch.values(),
    ]
    for i, left in enumerate(spans):
        for right in spans[i + 1 :]:
            if max(left.first, right.first) <= min(left.last, right.last):
                assert max(left.offset, right.offset) >= min(
                    left.offset + left.size, right.offset + right.size
                )
    if name == "views":
        result = layout.buffers[planned.case.graph._outputs["result"]]
        assert result.storage in {"alias", "input"}
    if name == "kv_cache":
        assert layout.cache_size == 2 * 8 * 2 * 8 * 4
