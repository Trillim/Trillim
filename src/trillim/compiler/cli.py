from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Sequence
from pathlib import Path

from .compiler_dataclasses import CompileArguments
from .compiler_pydantic_models import CompileRequest


def _checkpoint(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ArgumentTypeError("expected NAME=PATH with a nonempty name and path")
    return name, Path(path)


def parse_args(args: Sequence[str] | None = None) -> CompileRequest:
    parser = ArgumentParser(prog="trillim compile")
    parser.add_argument("tdl_path", type=Path)
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        type=_checkpoint,
        metavar="NAME=PATH",
        help="bind a checkpoint name to a path; repeat for additional checkpoints",
    )
    arguments = parser.parse_args(args, namespace=CompileArguments())
    return CompileRequest(
        tdl_path=arguments.tdl_path,
        checkpoints=dict(arguments.checkpoints),
    )
