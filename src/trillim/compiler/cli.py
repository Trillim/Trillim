from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from .compiler_dataclasses import CompileArguments
from .compiler_pydantic_models import CompileRequest


def _checkpoint(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ArgumentTypeError("expected NAME=PATH with a nonempty name and path")
    return name, Path(path)


def _shape(value: str) -> tuple[str, tuple[int, ...]]:
    name, separator, dimensions = value.partition("=")
    try:
        shape = tuple(int(item) for item in dimensions.split(",")) if dimensions else ()
    except ValueError as error:
        raise ArgumentTypeError(
            "expected NAME=DIM,DIM with integer dimensions"
        ) from error
    if not separator or not name or any(item < 0 for item in shape):
        raise ArgumentTypeError("expected NAME=DIM,DIM with nonnegative dimensions")
    return name, shape


def _dtype(value: str) -> tuple[str, str]:
    name, separator, dtype = value.partition("=")
    if not separator or not name or not dtype:
        raise ArgumentTypeError("expected NAME=DTYPE")
    return name, dtype


def _value_bound(value: str) -> tuple[str, tuple[int, int]]:
    name, separator, text = value.partition("=")
    try:
        bounds = tuple(int(item) for item in text.split(","))
    except ValueError as error:
        raise ArgumentTypeError("expected NAME=MIN,MAX") from error
    if not separator or not name or len(bounds) != 2:
        raise ArgumentTypeError("expected NAME=MIN,MAX")
    if bounds[0] > bounds[1]:
        raise ArgumentTypeError("expected NAME=MIN,MAX with MIN <= MAX")
    return name, (bounds[0], bounds[1])


def parse_args(args: Sequence[str] | None = None) -> CompileRequest:
    parser = ArgumentParser(prog="trillim compile")
    parser.add_argument("tdl_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build/model"))
    parser.add_argument("--backend", choices=("scalar", "avx2"), default="scalar")
    parser.add_argument(
        "--bound", dest="bounds", action="append", type=_shape, default=[]
    )
    parser.add_argument(
        "--value-bound",
        dest="value_bounds",
        action="append",
        type=_value_bound,
        default=[],
    )
    parser.add_argument(
        "--dtype", dest="dtypes", action="append", type=_dtype, default=[]
    )
    parser.add_argument(
        "--depth", choices=("none", "low", "medium", "high"), default="none"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-budget", type=int, default=8 * 1024**3)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--search-seconds", type=float, default=10)
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        type=_checkpoint,
        metavar="NAME=PATH",
        help="bind a checkpoint name to a path; repeat for additional checkpoints",
    )
    arguments = parser.parse_args(args, namespace=CompileArguments())
    checkpoints: dict[str, Path] = {}
    for name, path in arguments.checkpoints:
        if name in checkpoints:
            parser.error(f"duplicate checkpoint name: {name}")
        checkpoints[name] = path
    try:
        return CompileRequest.model_validate(
            {
                "tdl_path": arguments.tdl_path,
                "checkpoints": checkpoints,
                "output": arguments.output,
                "backend": arguments.backend,
                "bounds": dict(arguments.bounds),
                "value_bounds": dict(arguments.value_bounds),
                "dtypes": dict(arguments.dtypes),
                "depth": arguments.depth,
                "threads": arguments.threads,
                "memory_budget": arguments.memory_budget,
                "beam_width": arguments.beam_width,
                "search_seconds": arguments.search_seconds,
            }
        )
    except ValidationError as error:
        detail = error.errors()[0]
        field = ".".join(str(item) for item in detail["loc"])
        parser.error(f"{field}: {detail['msg']}")
