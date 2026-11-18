from __future__ import annotations

from collections.abc import Callable

import pytest

from trillim.compiler.kernels.kernel_dataclasses import _Kernel

from .compiler_dataclasses import PlannedCase


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
