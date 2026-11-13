"""Measure matched teacher-forced CPU decoder calls on an existing Trillim bundle."""

from __future__ import annotations

import argparse
import hashlib
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trillim.kernels.neon.benchmark_models import ManifestSymbols
from trillim.runtime import Runtime


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    bundle: Path
    tokens_file: Path
    output: Path
    prompt: int = Field(default=16, ge=1)
    decode: int = Field(default=128, ge=1)
    cache: int = Field(default=512, ge=1)
    threads: int = Field(default=20, ge=1)
    warmups: int = Field(default=2, ge=0)
    repeats: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_capacity(self) -> Options:
        if self.prompt + self.decode > self.cache:
            raise ValueError("cache capacity must fit prompt and decode calls")
        return self


class Session(BaseModel):
    warmup: bool
    prefill_seconds: float
    decode_seconds: list[float]
    predicted_ids: list[int]
    prefill_sha256: str


class Results(BaseModel):
    engine: str = "Trillim"
    device: str = "cpu"
    options: Options
    host: str
    repository_revision: str
    kernel_library_sha256: dict[str, str]
    kernel_symbols: list[str]
    executable_sha256: str
    teacher_sha256: str
    logits_path: Path
    sessions: list[Session]
    prefill_median_seconds: float
    decode_median_tokens_per_second: float
    decode_aggregate_tokens_per_second: float
    timing_scope: str


def benchmark(options: Options) -> Results:
    teacher = np.asarray(
        [int(value) for value in options.tokens_file.read_text().split()], dtype=np.int64
    )
    if len(teacher) < options.prompt + options.decode:
        raise ValueError("teacher file must contain at least prompt + decode tokens")
    if bool(np.any(teacher < 0)):
        raise ValueError("teacher tokens must be nonnegative")
    manifest_bytes = (options.bundle / "executable.json").read_bytes()
    manifest = ManifestSymbols.model_validate_json(manifest_bytes)
    libraries = sorted({kernel.library for kernel in manifest.kernels.values()})
    hashes = {
        name: hashlib.sha256((options.bundle / name).read_bytes()).hexdigest()
        for name in libraries
    }
    sessions: list[Session] = []
    first_logits: list[np.ndarray[tuple[int, ...], np.dtype[np.float32]]] = []
    for repeat in range(options.warmups + options.repeats):
        with Runtime(options.bundle, threads=options.threads) as runtime:
            prompt = teacher[: options.prompt]
            positions = np.arange(options.prompt, dtype=np.int64)
            started = time.perf_counter()
            logits = runtime.run({"tokens": prompt, "positions": positions})[
                "logits"
            ].ravel()
            prefill = time.perf_counter() - started
            if not bool(np.isfinite(logits).all()):
                raise ValueError("prefill logits must be finite")
            prefill_hash = hashlib.sha256(logits.tobytes()).hexdigest()
            predicted = [int(np.argmax(logits))]
            if repeat == options.warmups:
                first_logits.append(np.asarray(logits, dtype=np.float32).copy())
            samples: list[float] = []
            for step in range(options.decode):
                offset = options.prompt + step
                token = teacher[offset : offset + 1]
                positions = np.asarray([offset], dtype=np.int64)
                started = time.perf_counter()
                logits = runtime.run({"tokens": token, "positions": positions})[
                    "logits"
                ].ravel()
                samples.append(time.perf_counter() - started)
                if not bool(np.isfinite(logits).all()):
                    raise ValueError("decode logits must be finite")
                predicted.append(int(np.argmax(logits)))
                if repeat == options.warmups and step < 4:
                    first_logits.append(np.asarray(logits, dtype=np.float32).copy())
            sessions.append(
                Session(
                    warmup=repeat < options.warmups,
                    prefill_seconds=prefill,
                    decode_seconds=samples,
                    predicted_ids=predicted,
                    prefill_sha256=prefill_hash,
                )
            )
        print(
            f"session {repeat + 1}: prefill {prefill:.6f}s; "
            f"median decode {1 / statistics.median(samples):.3f} tok/s",
            flush=True,
        )
    measured = [session for session in sessions if not session.warmup]
    decode = [elapsed for session in measured for elapsed in session.decode_seconds]
    options.output.parent.mkdir(parents=True, exist_ok=True)
    logits_path = options.output.with_name(options.output.name + "-first5-logits.npy")
    np.save(logits_path, np.stack(first_logits))
    return Results(
        options=options,
        host=platform.platform(),
        repository_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        kernel_library_sha256=hashes,
        kernel_symbols=sorted({kernel.symbol for kernel in manifest.kernels.values()}),
        executable_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        teacher_sha256=hashlib.sha256(teacher.tobytes()).hexdigest(),
        logits_path=logits_path,
        sessions=sessions,
        prefill_median_seconds=statistics.median(
            session.prefill_seconds for session in measured
        ),
        decode_median_tokens_per_second=1 / statistics.median(decode),
        decode_aggregate_tokens_per_second=len(decode) / sum(decode),
        timing_scope="CPU Runtime.run including input binding, execution and owned output copy; excludes loading, tokenization, input array preparation, argmax and saving. Shared teacher stream; FP32 activations/cache and native checkpoint weight values. Cache option documents the compiled TDL capacity; this helper does not change it.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--tokens-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=int, default=16)
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--cache", type=int, default=512)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    options = Options.model_validate(vars(parser.parse_args()))
    results = benchmark(options)
    options.output.with_suffix(".json").write_text(
        results.model_dump_json(indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
