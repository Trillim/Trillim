from __future__ import annotations

import tarfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import TypeAdapter

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def offline_processing() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("TRANSFORMERS_OFFLINE", "1")
        yield


@pytest.fixture(scope="session")
def assets(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("processing-assets")
    directories = {"pocket": FIXTURES / "pocket"}
    for name in ("qwen3", "whisper"):
        directory = root / name
        directory.mkdir()
        with tarfile.open(FIXTURES / f"{name}.tar.gz") as archive:
            archive.extractall(directory, filter="data")
        directories[name] = directory
    return directories


@pytest.fixture(scope="session")
def expected() -> dict[str, list[int] | list[float] | str]:
    # Token IDs come from the bundled original tokenizers; schedules from FLUX.2.
    return TypeAdapter(dict[str, list[int] | list[float] | str]).validate_json(
        (FIXTURES / "expected.json").read_bytes()
    )


@pytest.fixture(scope="session")
def audio_reference() -> dict[str, NDArray[np.float32]]:
    # The real speech clip's goldens use OpenAI Whisper and NeMo's PyTorch formulas.
    with np.load(FIXTURES / "expected_audio.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    for array in arrays.values():
        array.flags.writeable = False
    return arrays
