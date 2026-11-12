"""Measure cancellation of a running decoder bundle through Runtime.cancel().

Use bundles produced by benchmark_models.py. Each sample opens a fresh session
and requests cancellation after --delay seconds. Timings include the Python
executor and native kernels; model loading and session shutdown are excluded.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from threading import Timer

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from trillim.runtime import CancelledError, Runtime


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    bundle: Path
    output: Path
    threads: int = Field(default=20, ge=1)
    prompt: int = Field(default=16, ge=1)
    delay: float = Field(default=0.02, gt=0, allow_inf_nan=False)
    repeats: int = Field(default=5, ge=1)


def request_cancellation(runtime: Runtime, requested: list[float]) -> None:
    requested.append(time.perf_counter())
    runtime.cancel()


def benchmark(options: Options) -> dict[str, object]:
    tokens = np.resize(np.array([9707, 11, 358, 1097], dtype=np.int64), options.prompt)
    positions = np.arange(options.prompt, dtype=np.int64)
    samples: list[dict[str, float | bool | None]] = []
    latencies: list[float] = []
    closed_latencies: list[float] = []
    for _ in range(options.repeats):
        requested: list[float] = []
        with Runtime(options.bundle, threads=options.threads) as runtime:
            timer = Timer(options.delay, request_cancellation, args=(runtime, requested))
            cancelled = False
            started = time.perf_counter()
            timer.start()
            try:
                runtime.run({"tokens": tokens, "positions": positions})
            except CancelledError:
                cancelled = True
            finally:
                finished = time.perf_counter()
                timer.cancel()
                timer.join()
        closed = time.perf_counter()
        latency = finished - requested[0] if cancelled and requested else None
        if cancelled and latency is not None:
            latencies.append(latency)
            closed_latencies.append(closed - requested[0])
        samples.append(
            {
                "cancelled": cancelled,
                "run_seconds": finished - started,
                "request_seconds": requested[0] - started if requested else None,
                "cancellation_latency_seconds": latency,
                "session_close_seconds": closed - finished,
                "request_to_session_closed_seconds": closed - requested[0]
                if cancelled and requested
                else None,
            }
        )
    return {
        "bundle": str(options.bundle),
        "host": platform.platform(),
        "threads": options.threads,
        "prompt_tokens": options.prompt,
        "requested_delay_seconds": options.delay,
        "samples": samples,
        "all_cancelled": len(latencies) == options.repeats,
        "median_cancellation_latency_seconds": statistics.median(latencies)
        if latencies
        else None,
        "maximum_cancellation_latency_seconds": max(latencies) if latencies else None,
        "maximum_request_to_session_closed_seconds": max(closed_latencies)
        if closed_latencies
        else None,
        "timing_scope": "Runtime.cancel request to Runtime.run raising CancelledError; includes Python executor and native kernels",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--prompt", type=int, default=16)
    parser.add_argument("--delay", type=float, default=0.02)
    parser.add_argument("--repeats", type=int, default=5)
    options = Options.model_validate(vars(parser.parse_args()))
    result = benchmark(options)
    options.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
