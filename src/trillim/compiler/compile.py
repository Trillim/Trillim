from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .. import _VERSION_ROOT
from .artifacts.artifact_pydantic_models import _ExecutableManifest
from .bench.tune import _tune
from .compiler_pydantic_models import _CompileRequest
from .kernels.catalog import _discover
from .kernels.kernel_dataclasses import _Backend
from .passes.allocate import _allocate
from .passes.emit import _emit
from .passes.infer import _infer
from .passes.match import _match, _selected_loop_inference
from .passes.optimize import _optimize
from .passes.packing import _set_packings
from .passes.processing import _processing
from .passes.schedule import _schedule
from .passes.select import _select
from .passes.trace import _trace
from .passes.validate import _validate
from .passes.verify import _verify
from .passes.walk import _collect_checkpoints
from .weights.bundle import _write_weights
from .weights.resolve import _resolve_payloads, _resolve_weights
from .weights.weights_dataclasses import _WeightRequest


def _compile_model(
    request: _CompileRequest, *, progress: Callable[[str], None] | None = None
) -> _ExecutableManifest:
    model_name = request.tdl_path.stem
    if len(request.checkpoints) == 1:
        checkpoint = next(iter(request.checkpoints.values())).resolve()
        model_name = checkpoint.parent.name if checkpoint.is_file() else checkpoint.name
    output = (request.output or _VERSION_ROOT / "compiled" / model_name).resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    try:
        report("Tracing and validating TDL")
        graph = _trace(request)
        _validate(graph)
        processing = _processing(graph, request.checkpoints, temporary)
        report("Resolving checkpoint tensors")
        assert graph._region is not None
        _collect_checkpoints(graph._region, request.checkpoints)
        weights = _resolve_weights(graph, request.checkpoints)
        report("Inferring shapes and dtypes")
        inference = _infer(
            graph,
            max_input_size=request.max_input_size,
        )
        report("Simplifying the graph")
        if _optimize(graph, inference):
            inference = _infer(
                graph,
                max_input_size=request.max_input_size,
            )
        _validate(graph)
        report("Loading and matching kernels")
        backends: tuple[_Backend, ...] = (
            ("scalar", request.backend)
            if request.backend != "scalar" and request.depth != "none"
            else (request.backend,)
        )
        kernels = _discover(backends)
        all_matches = _match(graph, kernels, inference, algebraic=False)
        structural = all_matches
        matches = replace(
            all_matches,
            candidates={
                operator: tuple(
                    candidate
                    for candidate in candidates
                    if candidate.kernel.backend == request.backend
                )
                for operator, candidates in all_matches.candidates.items()
            },
        )
        selection = _select(
            graph,
            matches,
            fast=True,
        )
        if request.depth == "high":
            report("Exploring algebraic fusion alternatives")
            all_matches = _match(graph, kernels, inference, algebraic=True)
            matches = replace(
                all_matches,
                candidates={
                    operator: tuple(
                        candidate
                        for candidate in candidates
                        if candidate.kernel.backend == request.backend
                    )
                    for operator, candidates in all_matches.candidates.items()
                },
            )
        report("Writing model weights")
        selection = _set_packings(graph, matches, selection, weights)
        representations = None
        if request.depth != "none":
            formats = {
                (binding.logical, binding.dtype, "canonical")
                for binding in weights.bindings.values()
            }
            for candidates in all_matches.candidates.values():
                for candidate in candidates:
                    for tensor, packing in candidate.packings.items():
                        binding = weights.bindings[tensor]
                        formats.add((binding.logical, binding.dtype, packing))
            representations = [
                _WeightRequest(logical, dtype, packing, f"{logical}@{dtype}@{packing}")
                for logical, dtype, packing in sorted(formats)
            ]
        manifest = _write_weights(
            _resolve_payloads(weights, representations), temporary / "weights.bin"
        )
        if request.depth != "none":
            report("Benchmarking representative kernel workloads")
        tuning = _tune(
            graph,
            all_matches,
            inference,
            weights,
            manifest,
            temporary / "weights.bin",
            request,
            selection,
            structural,
        )
        selection = tuning.selection
        if request.depth != "none":
            report(
                f"Measured {tuning.measured} fresh kernel workloads; "
                f"accepted {tuning.accepted} region improvements "
                f"({tuning.propagated} propagated to equivalent regions)"
            )
        report("Scheduling selected kernels")
        selection = _set_packings(graph, matches, selection, weights)
        selected_inference = _selected_loop_inference(inference, selection)
        _schedule(graph, selected_inference, selection)
        report("Allocating intermediate and cache memory")
        layout = _allocate(
            graph,
            selected_inference,
            selection,
            weights,
            manifest,
            transfer_kernel=next(
                (
                    kernel
                    for kernel in kernels
                    if kernel.name.endswith("_copy")
                    and kernel.backend == request.backend
                ),
                None,
            ),
        )
        report("Writing and verifying executable manifest")
        executable = _emit(graph, layout, selection, request, temporary)
        workload = next(
            (
                {"input": name, "axis": axis}
                for name, tensor in graph._inputs.items()
                if tensor.shape is not None
                for axis, dimension in enumerate(tensor.shape.dimensions)
                if dimension is None
            ),
            None,
        )
        executable = executable.model_copy(
            update={"processing": processing, "workload": workload}
        )
        (temporary / "executable.json").write_text(executable.model_dump_json() + "\n")
        _verify(temporary)
        if output.exists():
            raise FileExistsError(
                f"output directory appeared during compilation: {output}"
            )
        temporary.rename(output)
        report(f"Compiled model written to {output}")
        return executable
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
