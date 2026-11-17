from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from trillim.compiler.kernels import build, catalog
from trillim.compiler.kernels.kernel_dataclasses import _Kernel
from trillim.compiler.kernels.kernel_pydantic_models import _KernelRequest

from .compiler_dataclasses import PlannedCase


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


@pytest.fixture(scope="session")
def model_plans(
    tmp_path_factory: pytest.TempPathFactory, scalar_kernels: tuple[_Kernel, ...]
) -> Callable[[str], PlannedCase]:
    from .planning import plan

    root = tmp_path_factory.mktemp("model-plans")
    cached: dict[str, PlannedCase] = {}

    def prepare(name: str) -> PlannedCase:
        if name not in cached:
            cached[name] = plan(name, root / name, scalar_kernels)
        return cached[name]

    return prepare
