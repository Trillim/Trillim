from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "src/trillim"
    sources = root / "backends/cpu"
    sys.path.insert(0, str(root.parent))
    from trillim.compiler.toolchain import _compilers

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
    failures: list[str] = []
    try:
        compilers = _compilers(cxx=True)
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
    status = 1
    for compiler in compilers:
        print(f"Building CPU worker with {' '.join(compiler)}", flush=True)
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
        except (subprocess.CalledProcessError, OSError) as error:
            failures.append(f"{' '.join(compiler)}: {error}")
            status = (
                error.returncode
                if isinstance(error, subprocess.CalledProcessError)
                else 1
            )
            continue
        print(f"Built CPU worker: {output}")
        return
    print("all C++ compilers failed:\n" + "\n".join(failures), file=sys.stderr)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
