from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

_VERSION_ROOT = (
    _Path(_os.environ.get("TRILLIM_HOME", "~/.trillim")).expanduser().resolve()
    / "1.0.0"
)


def _main() -> None:
    from .compiler.cli import _parse_args
    from .compiler.compile import _compile_model

    arguments = _sys.argv[1:]
    if arguments and arguments[0] == "compile":
        arguments = arguments[1:]
    try:
        _compile_model(_parse_args(arguments), progress=print)
    except KeyboardInterrupt:
        print("Compilation cancelled.", file=_sys.stderr)
        raise SystemExit(130) from None
