from __future__ import annotations

import json
import signal
import sys
from argparse import ArgumentParser
from pathlib import Path

import soundfile as sf  # type: ignore[import-untyped]
from PIL import Image
from pydantic import JsonValue, TypeAdapter

from ..processing import Audio
from .model import Model
from .runtime_dataclasses import _RunArguments

_INPUT = TypeAdapter[JsonValue](JsonValue)


def _run(arguments: list[str]) -> None:
    parser = ArgumentParser(prog="trillim run")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        help="audio/image output path; numbered for multiple requests",
    )
    options = _RunArguments()
    parser.parse_args(arguments, namespace=options)

    def terminate(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    def write_text(chunk: str) -> None:
        print(chunk, end="", flush=True)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        with Model[object, object](options.directory, threads=options.threads) as model:
            on_text = write_text if model.supports_text_streaming else None
            index = 0
            for line in sys.stdin:
                if not line.strip():
                    continue
                value: object = (
                    _INPUT.validate_json(line)
                    if line.lstrip().startswith(("{", "[", '"'))
                    else line.rstrip("\r\n")
                )
                result = model.run(value, on_text=on_text)
                if isinstance(result, (Audio, Image.Image)):
                    path = options.output or Path(
                        "output.wav" if isinstance(result, Audio) else "output.png"
                    )
                    if index:
                        path = path.with_stem(f"{path.stem}-{index}")
                    if isinstance(result, Audio):
                        sf.write(str(path), result.samples, result.sample_rate)
                    else:
                        result.save(path)
                    print(path, flush=True)
                elif isinstance(result, str):
                    print(result if on_text is None else "", flush=True)
                else:
                    print(
                        json.dumps(result, separators=(",", ":"), allow_nan=False),
                        flush=True,
                    )
                index += 1
    finally:
        signal.signal(signal.SIGTERM, previous)
