from __future__ import annotations

import json
import signal
import sys
from argparse import ArgumentParser
from pathlib import Path

from pydantic import ConfigDict, JsonValue, TypeAdapter

from .runtime_dataclasses import _RunArguments
from .session import Runtime

_INPUTS = TypeAdapter(dict[str, JsonValue], config=ConfigDict(strict=True))


def _run(arguments: list[str]) -> None:
    parser = ArgumentParser(prog="trillim run")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--threads", type=int)
    options = _RunArguments()
    parser.parse_args(arguments, namespace=options)

    def terminate(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    with Runtime(options.directory, threads=options.threads) as runtime:
        for line in sys.stdin:
            if line.strip():
                inputs = _INPUTS.validate_json(line)
                outputs = runtime.run(inputs)
                print(
                    json.dumps(
                        {name: value.tolist() for name, value in outputs.items()},
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    flush=True,
                )
