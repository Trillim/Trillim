from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .. import _VERSION_ROOT
from .artifacts.artifact_pydantic_models import _ExecutableManifest, _WorkloadPlan
from .bench.tune import _tune
from .compiler_pydantic_models import _CompileRequest
from .kernels.catalog import _discover
from .kernels.kernel_dataclasses import _Backend
from .passes.allocate import _allocate
from .passes.emit import _emit
from .passes.infer import _infer
from .passes.match import _match
from .passes.optimize import _optimize
from .passes.schedule import _schedule
from .passes.select import _select
from .passes.trace import _trace
from .passes.validate import _validate
from .passes.verify import _verify
from .passes.walk import _collect_checkpoints
from .weights.bundle import _write_weights
from .weights.resolve import _resolve_weights


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
        _schedule(graph, inference)
        report("Loading and matching kernels")
        backends: tuple[_Backend, ...] = (
            ("scalar", request.backend)
            if request.backend != "scalar" and request.depth != "none"
            else (request.backend,)
        )
        kernels = _discover(backends)
        all_matches = _match(graph, kernels, inference)
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
        if request.depth == "low":
            all_matches = replace(
                all_matches,
                candidates={
                    operator: tuple(
                        candidate
                        for candidate in candidates
                        if candidate.covered == selection.calls[operator].covered
                    )
                    for operator, candidates in all_matches.candidates.items()
                    if operator in selection.calls
                },
            )
            matches = replace(
                matches,
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
        manifest = _write_weights(weights, temporary / "weights.bin")
        if request.depth != "none":
            report("Benchmarking representative kernel workloads")
        tuning = _tune(
            all_matches,
            inference,
            weights,
            manifest,
            temporary / "weights.bin",
            request,
        )
        if tuning.costs:
            selection = _select(
                graph,
                matches,
                costs=tuning.costs,
                fast=request.depth != "high",
            )
        report("Allocating intermediate and cache memory")
        layout = _allocate(
            graph,
            inference,
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
        variants: list[_WorkloadPlan] = []
        baseline = tuple(
            (id(operator), call.kernel.name)
            for operator, call in selection.calls.items()
        )
        all_kernels = dict(executable.kernels)
        arena_size = executable.arena_size
        previous = 0
        for point, costs in tuning.per_workload.items():
            chosen = _select(
                graph,
                matches,
                costs=costs,
                fast=request.depth != "high",
            )
            fingerprint = tuple(
                (id(operator), call.kernel.name)
                for operator, call in chosen.calls.items()
            )
            if costs and workload is not None and fingerprint != baseline:
                variant_layout = _allocate(
                    graph,
                    inference,
                    chosen,
                    weights,
                    manifest,
                    transfer_kernel=layout.transfer_kernel,
                )
                variant = _emit(graph, variant_layout, chosen, request, temporary)
                dump = variant.model_dump(mode="json")
                variants.append(
                    _WorkloadPlan.model_validate(
                        {
                            "minimum_tokens": previous + 1,
                            "maximum_tokens": point,
                            "buffers": dump["buffers"],
                            "program": dump["program"],
                            "expressions": dump["expressions"],
                            "arena_size": variant.arena_size,
                        }
                    )
                )
                all_kernels.update(variant.kernels)
                arena_size = max(arena_size, variant.arena_size)
            previous = point
        executable = executable.model_copy(
            update={
                "workloads": tuning.workloads,
                "workload": workload,
                "plans": variants,
                "kernels": all_kernels,
                "arena_size": arena_size,
            }
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
