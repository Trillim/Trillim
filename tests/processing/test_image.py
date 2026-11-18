from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from PIL import Image
from pydantic import ValidationError

from trillim.processing import postprocessing, preprocessing
from trillim.processing.utils import image_from_tensor

FIXTURES = Path(__file__).parent / "fixtures"
PROMPT = "A red kite over a green field."


@pytest.mark.parametrize(
    "rows,columns,steps",
    [(64, 64, 4), (32, 64, 8), (120, 120, 50), (1, 1, 1), (1, 1, 1000)],
)
def test_flux_inputs_and_schedule(
    assets: dict[str, Path],
    expected: dict[str, list[int] | list[float] | str],
    rows: int,
    columns: int,
    steps: int,
) -> None:
    request = {
        "prompt": PROMPT,
        "height": rows * 16,
        "width": columns * 16,
        "seed": 2**64 - 1,
        "steps": steps,
    }
    inputs = cast(
        dict[str, NDArray[np.generic]],
        preprocessing.image(
            request,
            assets={"tokenizer": assets["qwen3"]},
            options={},
        ),
    )
    assert set(inputs) == {"tokens", "coordinates", "sigmas", "height", "width", "seed"}
    np.testing.assert_array_equal(inputs["tokens"], expected["chat_thinking_false"])
    assert inputs["tokens"].dtype == inputs["coordinates"].dtype == np.int32
    count = inputs["tokens"].size
    assert inputs["coordinates"].shape == (count + rows * columns, 4)
    np.testing.assert_array_equal(
        inputs["coordinates"][:count], [[0, 0, 0, i] for i in range(count)]
    )
    np.testing.assert_array_equal(
        inputs["coordinates"][count:],
        [[0, row, column, 0] for row in range(rows) for column in range(columns)],
    )
    assert inputs["sigmas"].dtype == np.float32
    np.testing.assert_allclose(
        inputs["sigmas"].astype(np.float32),
        np.asarray(expected[f"sigmas_{rows}_{columns}_{steps}"], dtype=np.float32),
        rtol=1e-6,
        atol=1e-7,
    )
    assert np.all(np.diff(inputs["sigmas"]) < 0)
    assert inputs["height"].shape == inputs["width"].shape == inputs["seed"].shape == ()
    assert inputs["height"].dtype == inputs["width"].dtype == np.int32
    assert inputs["height"].item() == request["height"]
    assert inputs["width"].item() == request["width"]
    assert inputs["seed"].dtype == np.uint64 and inputs["seed"].item() == 2**64 - 1


def test_flux_defaults(
    assets: dict[str, Path], expected: dict[str, list[int] | list[float] | str]
) -> None:
    inputs = preprocessing.image(
        PROMPT, assets={"tokenizer": assets["qwen3"]}, options={}
    )
    np.testing.assert_array_equal(inputs["height"], np.asarray(1024, dtype=np.int32))
    np.testing.assert_array_equal(inputs["width"], np.asarray(1024, dtype=np.int32))
    np.testing.assert_array_equal(inputs["seed"], np.asarray(0, dtype=np.uint64))
    np.testing.assert_allclose(
        np.asarray(inputs["sigmas"], dtype=np.float32),
        np.asarray(expected["sigmas_64_64_4"], dtype=np.float32),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("height", 0),
        ("height", 1936),
        ("width", 17),
        ("width", 1921),
        ("seed", -1),
        ("seed", 2**64),
        ("seed", True),
        ("steps", 0),
        ("steps", 1001),
        ("steps", 4.0),
        ("unknown", 1),
    ],
)
def test_flux_invalid_request(
    assets: dict[str, Path], field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        preprocessing.image(
            {"prompt": PROMPT, field: value},
            assets={"tokenizer": assets["qwen3"]},
            options={},
        )


def test_flux_prompt_limit(assets: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="exceeds 512 tokens"):
        preprocessing.image(
            "kite " * 600, assets={"tokenizer": assets["qwen3"]}, options={}
        )


@settings(max_examples=30, deadline=None)
@given(
    row=st.integers(0, 63),
    column=st.integers(0, 47),
    height=st.integers(1, 64),
    width=st.integers(1, 48),
)
def test_real_rgb_pixels_survive_model_tensor_round_trip(
    row: int, column: int, height: int, width: int
) -> None:
    with Image.open(FIXTURES / "portrait.png") as image:
        pixels = np.asarray(image)[row : row + height, column : column + width].copy()
    tensor = pixels.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1
    before = tensor.copy()
    decoded = image_from_tensor(tensor)
    assert decoded.mode == "RGB" and decoded.size == (pixels.shape[1], pixels.shape[0])
    np.testing.assert_array_equal(np.asarray(decoded), pixels)
    np.testing.assert_array_equal(tensor, before)
    result = postprocessing.image({"image": tensor}, assets={}, options={})
    np.testing.assert_array_equal(np.asarray(result), pixels)


def test_image_clips_model_range() -> None:
    with Image.open(FIXTURES / "portrait.png") as image:
        tensor = np.asarray(image).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1
    tensor[:, 0, 0] = [-2, 0, 2]
    np.testing.assert_array_equal(
        np.asarray(image_from_tensor(tensor))[0, 0], [0, 128, 255]
    )


@pytest.mark.parametrize("damage", ["channels", "rank", "nan", "inf"])
def test_image_rejects_invalid_tensors(damage: str) -> None:
    with Image.open(FIXTURES / "portrait.png") as image:
        tensor = np.asarray(image).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1
    if damage == "channels":
        tensor = tensor[:1]
    elif damage == "rank":
        tensor = tensor[None, :]
    else:
        tensor[0, 0, 0] = np.nan if damage == "nan" else np.inf
    with pytest.raises(ValueError, match="finite.*RGB"):
        image_from_tensor(tensor)
