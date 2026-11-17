from __future__ import annotations

from pathlib import Path

from trillim.compiler.artifacts.artifact_pydantic_models import _ExecutableManifest
from trillim.compiler.compiler_pydantic_models import _CompileRequest
from trillim.compiler.kernels.kernel_dataclasses import _Kernel
from trillim.compiler.passes.allocate import _allocate
from trillim.compiler.passes.emit import _emit
from trillim.compiler.passes.infer import _infer
from trillim.compiler.passes.match import _match
from trillim.compiler.passes.optimize import _optimize
from trillim.compiler.passes.schedule import _schedule
from trillim.compiler.passes.select import _select
from trillim.compiler.passes.validate import _validate
from trillim.compiler.weights.bundle import _write_weights
from trillim.compiler.weights.resolve import _resolve_payloads, _resolve_weights

from .compiler_dataclasses import PlannedCase
from .model_cases import FIXTURES, model_case


def plan(name: str, directory: Path, kernels: tuple[_Kernel, ...]) -> PlannedCase:
    case = model_case(name)
    graph = case.graph
    _validate(graph)
    weights = _resolve_weights(
        graph, {"checkpoint": FIXTURES / "Qwen3-0.6B.safetensors"}
    )
    inference = _infer(graph, max_input_size=4)
    if _optimize(graph, inference):
        inference = _infer(graph, max_input_size=4)
    selected = _select(graph, _match(graph, kernels, inference), fast=True)
    manifest = _write_weights(_resolve_payloads(weights), directory / "weights.bin")
    _schedule(graph, inference, selected)
    layout = _allocate(
        graph,
        inference,
        selected,
        weights,
        manifest,
        transfer_kernel=next(
            kernel for kernel in kernels if kernel.name == "trillim_scalar_copy"
        ),
    )
    return PlannedCase(case, inference, selected, layout, directory)


def emit(planned: PlannedCase) -> _ExecutableManifest:
    request = _CompileRequest(
        tdl_path=Path(__file__), backend="scalar", max_input_size=4, threads=1
    )
    return _emit(
        planned.case.graph,
        planned.layout,
        planned.selection,
        request,
        planned.directory,
    )
