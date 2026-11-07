from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import monotonic

from .artifacts.artifact_pydantic_models import ExecutableManifest, WorkloadPlan
from .bench.tune import tune
from .compiler_pydantic_models import CompileRequest
from .kernels.catalog import discover
from .kernels.kernel_dataclasses import Backend
from .passes.allocate import allocate
from .passes.emit import emit
from .passes.infer import infer
from .passes.match import match
from .passes.optimize import optimize
from .passes.schedule import schedule
from .passes.select import select
from .passes.trace import trace
from .passes.validate import validate
from .passes.verify import verify
from .weights import resolve_weights, write_weights


def compile_model(
    request: CompileRequest, *, progress: Callable[[str], None] | None = None
) -> ExecutableManifest:
    output = request.output.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    try:
        report("Tracing and validating TDL")
        graph = trace(request)
        validate(graph)
        report("Resolving checkpoint tensors")
        weights = resolve_weights(graph, request.checkpoints, dtypes=request.dtypes)
        report("Inferring shapes and dtypes")
        inference = infer(
            graph,
            bounds=request.bounds,
            value_bounds=request.value_bounds,
            dtypes=request.dtypes,
        )
        report("Simplifying the graph")
        if optimize(graph, inference):
            inference = infer(
                graph,
                bounds=request.bounds,
                value_bounds=request.value_bounds,
                dtypes=request.dtypes,
            )
        validate(graph)
        schedule(graph, inference)
        report("Building and matching kernels")
        backends: tuple[Backend, ...] = (
            ("scalar", request.backend)
            if request.backend != "scalar" and request.depth != "none"
            else (request.backend,)
        )
        kernels = discover(backends)
        all_matches = match(graph, kernels, inference)
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
        selection = select(
            graph,
            matches,
            beam_width=request.beam_width,
            search_seconds=request.search_seconds,
            fast=True,
        )
        report("Writing model weights")
        manifest = write_weights(weights, temporary / "weights.bin")
        if request.depth != "none":
            report("Benchmarking representative kernel workloads")
        deadline = monotonic() + request.search_seconds
        tuning_request = request.model_copy(
            update={"search_seconds": request.search_seconds * 0.8}
        )
        tuning = tune(
            all_matches,
            inference,
            weights,
            manifest,
            temporary / "weights.bin",
            tuning_request,
        )
        if tuning.costs:
            selection = select(
                graph,
                matches,
                beam_width=request.beam_width,
                search_seconds=max(0.0, deadline - monotonic()),
                costs=tuning.costs,
            )
        report("Allocating intermediate and cache memory")
        layout = allocate(
            graph,
            inference,
            selection,
            weights,
            manifest,
            memory_budget=request.memory_budget,
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
        executable = emit(graph, layout, selection, request, temporary)
        workload = next(
            (
                {"input": name, "axis": axis}
                for name, tensor in graph.inputs.items()
                if tensor.shape is not None
                for axis, dimension in enumerate(tensor.shape.dimensions)
                if dimension is None
            ),
            None,
        )
        variants: list[WorkloadPlan] = []
        baseline = tuple(
            (id(operator), call.kernel.name)
            for operator, call in selection.calls.items()
        )
        all_kernels = dict(executable.kernels)
        arena_size = executable.arena_size
        previous = 0
        for point, costs in tuning.per_workload.items():
            chosen = select(
                graph,
                matches,
                beam_width=request.beam_width,
                search_seconds=max(0.0, deadline - monotonic()),
                costs=costs,
            )
            fingerprint = tuple(
                (id(operator), call.kernel.name)
                for operator, call in chosen.calls.items()
            )
            if costs and workload is not None and fingerprint != baseline:
                variant_layout = allocate(
                    graph,
                    inference,
                    chosen,
                    weights,
                    manifest,
                    memory_budget=request.memory_budget,
                    transfer_kernel=layout.transfer_kernel,
                )
                variant = emit(graph, variant_layout, chosen, request, temporary)
                dump = variant.model_dump(mode="json")
                variants.append(
                    WorkloadPlan.model_validate(
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
        verify(temporary)
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
