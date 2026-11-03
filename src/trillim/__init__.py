from __future__ import annotations

import sys


def main() -> None:
    from .compiler import compile_model
    from .compiler.cli import parse_args

    arguments = sys.argv[1:]
    if arguments and arguments[0] == "compile":
        arguments = arguments[1:]
    try:
        compile_model(parse_args(arguments), progress=print)
    except KeyboardInterrupt:
        print("Compilation cancelled.", file=sys.stderr)
        raise SystemExit(130) from None
