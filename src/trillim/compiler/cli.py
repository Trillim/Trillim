from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from .compiler_dataclasses import _CompileArguments, _KernelArguments
from .compiler_pydantic_models import _CompileRequest
from .kernels.kernel_pydantic_models import _KernelRequest


def _checkpoint(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ArgumentTypeError("expected NAME=PATH with a nonempty name and path")
    return name, Path(path)


def _parse_args(args: Sequence[str] | None = None) -> _CompileRequest:
    parser = ArgumentParser(prog="trillim compile")
    parser.add_argument("tdl_path", type=Path)
    parser.add_argument(
        "checkpoints",
        type=_checkpoint,
        nargs="*",
        metavar="NAME=PATH",
        help="bind each TDL checkpoint name to its weights file or directory",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("scalar", "x86", "neon"))
    parser.add_argument("--depth", choices=("none", "low", "high"), default="none")
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--max-input-size",
        type=int,
        default=4096,
        help="maximum length of each variable input dimension (default: 4096)",
    )
    arguments = parser.parse_intermixed_args(args, namespace=_CompileArguments())
    checkpoints: dict[str, Path] = {}
    for name, path in arguments.checkpoints:
        if name in checkpoints:
            parser.error(f"duplicate checkpoint name: {name}")
        checkpoints[name] = path
    values: dict[str, object] = {
        "tdl_path": arguments.tdl_path,
        "checkpoints": checkpoints,
        "output": arguments.output,
        "depth": arguments.depth,
        "max_input_size": arguments.max_input_size,
    }
    if arguments.backend is not None:
        values["backend"] = arguments.backend
    if arguments.threads is not None:
        values["threads"] = arguments.threads
    try:
        return _CompileRequest.model_validate(values)
    except ValidationError as error:
        detail = error.errors()[0]
        field = ".".join(str(item) for item in detail["loc"])
        parser.error(f"{field}: {detail['msg']}")


def _parse_kernel_args(args: Sequence[str] | None = None) -> _KernelRequest:
    parser = ArgumentParser(prog="trillim kernels")
    parser.add_argument(
        "sources",
        type=Path,
        nargs="*",
        metavar="SOURCE.c",
        help="update these kernels; omit to rebuild all built-in kernels",
    )
    parser.add_argument("--backend", choices=("scalar", "x86", "neon"))
    arguments = parser.parse_intermixed_args(args, namespace=_KernelArguments())
    try:
        return _KernelRequest.model_validate(
            {"sources": tuple(arguments.sources), "backend": arguments.backend}
        )
    except ValidationError as error:
        parser.error(error.errors()[0]["msg"])
