from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from textwrap import dedent
from typing import cast

import numpy as np
import pytest

from trillim.compiler.compile import _compile_model
from trillim.compiler.compiler_pydantic_models import _CompileRequest
from trillim.compiler.kernels import build
from trillim.compiler.kernels.kernel_dataclasses import _Backend, _Kernel
from trillim.compiler.kernels.kernel_pydantic_models import _KernelRequest
from trillim.compiler.weights.reader import _load_checkpoint
from trillim.compiler.weights.resolve import _read_values

from .runtime_dataclasses import ModelArtifact


@pytest.fixture
def compile_tdl(
    scalar_kernels: tuple[_Kernel, ...], tmp_path: Path
) -> Callable[[str], Path]:
    def compile_source(source: str) -> Path:
        model = tmp_path / "model.tdl"
        model.write_text(dedent(source))
        directory = tmp_path / "bundle"
        _compile_model(
            _CompileRequest(
                tdl_path=model, output=directory, backend="scalar", threads=1
            )
        )
        return directory

    return compile_source


@pytest.fixture(scope="session")
def token_model(
    scalar_kernels: tuple[_Kernel, ...], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    directory = tmp_path_factory.mktemp("token-model") / "bundle"
    _compile_model(
        _CompileRequest(
            tdl_path=Path(__file__).parent / "fixtures" / "token_model.tdl",
            output=directory,
            backend="scalar",
            threads=1,
        )
    )
    return directory


@pytest.fixture(
    scope="session", params=tuple(dict.fromkeys(("scalar", build._host_backend())))
)
def qwen_layer(
    request: pytest.FixtureRequest,
    scalar_kernels: tuple[_Kernel, ...],
    tmp_path_factory: pytest.TempPathFactory,
) -> ModelArtifact:
    backend = cast(_Backend, request.param)
    if backend != "scalar":
        build._build(_KernelRequest(backend=backend), progress=lambda message: None)
    checkpoint_path = Path(__file__).parent / "fixtures" / "qwen3-layer0.safetensors"
    checkpoint = _load_checkpoint("checkpoint", checkpoint_path)
    embedding = checkpoint.tensors["model.embed_tokens.weight"]
    assert embedding.shape[1] == 1024
    values = _read_values(embedding, 0, 8 * 1024).reshape(8, 1024).astype(np.float32)
    directory = tmp_path_factory.mktemp(f"qwen-{backend}") / "compiled"
    _compile_model(
        _CompileRequest(
            tdl_path=Path(__file__).parent / "fixtures" / "qwen3.tdl",
            checkpoints={"checkpoint": checkpoint_path},
            output=directory,
            backend=backend,
            max_input_size=512,
            threads=2,
        )
    )
    return ModelArtifact(directory, values)
