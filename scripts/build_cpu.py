from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "src/trillim"
    sources = root / "backends/cpu"
    compiler = shlex.split(os.environ.get("CXX", "c++"), posix=sys.platform != "win32")
    if sys.platform == "win32":
        compiler = [
            argument.removeprefix('"').removesuffix('"') for argument in compiler
        ]
    if not compiler:
        raise SystemExit("CXX must name a C++20 compiler")
    flags = [
        "-O3",
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wpedantic",
        "-ffp-contract=off",
        "-municode" if sys.platform == "win32" else "-pthread",
    ]
    libraries = ["-ldl"] if sys.platform == "linux" else []
    home = Path(os.environ.get("TRILLIM_HOME", "~/.trillim")).expanduser().resolve()
    name = "worker.exe" if sys.platform == "win32" else "worker"
    output = home / "1.0.0/backends/cpu" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(prefix=".build-", dir=output.parent) as temporary:
            executable = Path(temporary) / name
            subprocess.run(
                [
                    *compiler,
                    *flags,
                    "-I",
                    str(root / "kernels"),
                    *(str(path) for path in sorted(sources.glob("*.cpp"))),
                    *libraries,
                    "-o",
                    str(executable),
                ],
                check=True,
            )
            executable.replace(output)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
    except OSError as error:
        raise SystemExit(str(error)) from None
    print(f"Built CPU worker: {output}")


if __name__ == "__main__":
    main()
