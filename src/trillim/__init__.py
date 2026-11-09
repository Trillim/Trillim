from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

_VERSION_ROOT = (
    _Path(_os.environ.get("TRILLIM_HOME", "~/.trillim")).expanduser().resolve()
    / "1.0.0"
)


def _main() -> None:
    from argparse import ArgumentParser

    from .compiler.cli import _parse_args, _parse_kernel_args
    from .compiler.compile import _compile_model
    from .compiler.kernels.build import _build

    arguments = _sys.argv[1:]
    if not arguments or arguments[0] not in {"compile", "kernels"}:
        parser = ArgumentParser(prog="trillim")
        parser.add_argument("command", choices=("compile", "kernels"))
        parser.parse_args(arguments)
        return
    try:
        if arguments[0] == "compile":
            _compile_model(_parse_args(arguments[1:]), progress=print)
        else:
            try:
                _build(_parse_kernel_args(arguments[1:]), progress=print)
            except (OSError, ValueError, RuntimeError) as error:
                print(f"Kernel compilation failed: {error}", file=_sys.stderr)
                raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("Compilation cancelled.", file=_sys.stderr)
        raise SystemExit(130) from None
