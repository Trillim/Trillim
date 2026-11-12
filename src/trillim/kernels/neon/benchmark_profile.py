"""Attribute decoder Runtime.run time to native kernel symbols.

This diagnostic temporarily instruments native dispatch in its own process.
Use benchmark_models.py for uninstrumented performance measurements.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from trillim.backends.cpu.cpu_dataclasses import _BoundCall
from trillim.backends.cpu.executor import _CPU
from trillim.runtime import Runtime


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    bundle: Path
    output: Path
    threads: int = Field(default=20, ge=1)
    prompt: int = Field(default=16, ge=1)
    decode: int = Field(default=4, ge=1)


def profile_run(
    runtime: Runtime, inputs: dict[str, NDArray[np.generic]]
) -> tuple[dict[str, NDArray[np.generic]], dict[str, object]]:
    timings: dict[str, list[float]] = defaultdict(list)
    original = _CPU._invoke_bound

    def measured(backend: _CPU, call: _BoundCall) -> None:
        started = time.perf_counter()
        try:
            original(backend, call)
        finally:
            timings[call.kernel].append(time.perf_counter() - started)

    started = time.perf_counter()
    with patch.object(_CPU, "_invoke_bound", measured):
        outputs = runtime.run(inputs)
    elapsed = time.perf_counter() - started
    native_seconds = sum(sum(samples) for samples in timings.values())
    rows = [
        {
            "kernel": name,
            "calls": len(samples),
            "seconds": sum(samples),
            "fraction_of_run": sum(samples) / elapsed,
            "mean_seconds": sum(samples) / len(samples),
        }
        for name, samples in sorted(
            timings.items(), key=lambda item: sum(item[1]), reverse=True
        )
    ]
    return outputs, {
        "run_seconds": elapsed,
        "native_seconds": native_seconds,
        "outside_native_seconds": elapsed - native_seconds,
        "native_calls": sum(len(samples) for samples in timings.values()),
        "kernels": rows,
    }


def benchmark(options: Options) -> dict[str, object]:
    prompt = np.resize(np.array([9707, 11, 358, 1097], dtype=np.int64), options.prompt)
    records: list[dict[str, object]] = []
    generated: list[int] = []
    with Runtime(options.bundle, threads=options.threads) as runtime:
        outputs, record = profile_run(
            runtime,
            {"tokens": prompt, "positions": np.arange(options.prompt, dtype=np.int64)},
        )
        record["phase"] = "prefill"
        records.append(record)
        generated.append(int(np.argmax(outputs["logits"])))
        for step in range(options.decode):
            outputs, record = profile_run(
                runtime,
                {
                    "tokens": np.array([generated[-1]], dtype=np.int64),
                    "positions": np.array([options.prompt + step], dtype=np.int64),
                },
            )
            record["phase"] = "decode"
            record["step"] = step
            records.append(record)
            generated.append(int(np.argmax(outputs["logits"])))
    return {
        "bundle": str(options.bundle),
        "threads": options.threads,
        "prompt_token_ids": [int(token) for token in prompt],
        "generated_token_ids": generated,
        "records": records,
        "timing_scope": "instrumented Runtime.run with per-symbol native-dispatch attribution; use uninstrumented model benchmark for final rates",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--prompt", type=int, default=16)
    parser.add_argument("--decode", type=int, default=4)
    options = Options.model_validate(vars(parser.parse_args()))
    result = benchmark(options)
    options.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
