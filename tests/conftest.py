from __future__ import annotations

from collections.abc import Iterator

import pytest

from trillim.compiler.kernels import build, catalog
from trillim.compiler.kernels.kernel_dataclasses import _Kernel
from trillim.compiler.kernels.kernel_pydantic_models import _KernelRequest


@pytest.fixture(scope="session")
def scalar_kernels(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[_Kernel, ...]]:
    root = tmp_path_factory.mktemp("scalar-kernels")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(build, "_VERSION_ROOT", root)
        patch.setattr(catalog, "_VERSION_ROOT", root)
        build._build(_KernelRequest(backend="scalar"), progress=lambda message: None)
        yield catalog._discover(("scalar",))
