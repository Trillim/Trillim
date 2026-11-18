from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from trillim.processing import postprocessing, preprocessing
from trillim.processing.utils import decode_tokens, tokenize


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("english", "Hello, world!"),
        ("unicode", "Café déjà vu — こんにちは世界"),
        ("spacing", "  def hello():\n    return 42\n"),
    ],
)
def test_qwen_plain_text(
    assets: dict[str, Path],
    expected: dict[str, list[int] | list[float] | str],
    name: str,
    text: str,
) -> None:
    tokens = tokenize(text, assets=assets["qwen3"], chat=False)
    assert tokens.dtype == np.int32
    np.testing.assert_array_equal(tokens, expected[f"plain_{name}"])
    assert decode_tokens(tokens, assets=assets["qwen3"]) == text
    inputs = preprocessing.text(
        {"text": text, "reset": True},
        assets={"tokenizer": assets["qwen3"]},
        options={"chat": False},
    )
    assert set(inputs) == {"tokens", "reset"}
    np.testing.assert_array_equal(inputs["tokens"], tokens)
    reset = inputs["reset"]
    assert isinstance(reset, np.ndarray)
    assert reset.shape == () and reset.dtype == np.bool_ and reset.item() is True


def test_qwen_chat_and_complete_history(
    assets: dict[str, Path], expected: dict[str, list[int] | list[float] | str]
) -> None:
    directory = assets["qwen3"]
    tokens = tokenize("Hello, world!", assets=directory)
    np.testing.assert_array_equal(tokens, expected["chat_default"])
    assert (
        decode_tokens(tokens, assets=directory, skip_special_tokens=False)
        == expected["chat_default_decoded"]
    )
    messages = [
        {"role": "system", "content": "Reply briefly."},
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "What is KV caching?"},
    ]
    request = {"messages": messages}
    before = deepcopy(request)
    inputs = preprocessing.text(request, assets={"tokenizer": directory}, options={})
    np.testing.assert_array_equal(inputs["tokens"], expected["conversation"])
    np.testing.assert_array_equal(inputs["reset"], np.asarray(False))
    assert request == before
    for thinking in (True, False):
        tokens = tokenize(
            "A red kite over a green field.", assets=directory, enable_thinking=thinking
        )
        np.testing.assert_array_equal(
            tokens, expected[f"chat_thinking_{str(thinking).lower()}"]
        )


def test_text_only_vision_bindings(
    assets: dict[str, Path], expected: dict[str, list[int] | list[float] | str]
) -> None:
    inputs = cast(
        dict[str, NDArray[np.generic]],
        preprocessing.text(
            "Hello, world!",
            assets={"tokenizer": assets["qwen3"]},
            options={"vision": True},
        ),
    )
    np.testing.assert_array_equal(inputs["tokens"], expected["chat_default"])
    length = inputs["tokens"].size
    assert set(inputs) == {
        "tokens",
        "reset",
        "vision.patches",
        "vision.coordinates",
        "vision.interpolation",
        "vision.segments",
        "vision.token_features",
        "positions",
        "rope_delta",
    }
    for name, shape, dtype in (
        ("vision.patches", (0, 3, 2, 16, 16), np.float32),
        ("vision.coordinates", (0, 2), np.int32),
        ("vision.interpolation", (0, 2), np.float32),
        ("vision.segments", (0,), np.int32),
    ):
        assert inputs[name].shape == shape and inputs[name].dtype == dtype
    np.testing.assert_array_equal(inputs["vision.token_features"], [-1] * length)
    np.testing.assert_array_equal(
        inputs["positions"], [[i, i, i] for i in range(length)]
    )
    assert inputs["rope_delta"].shape == () and inputs["rope_delta"].item() == 0


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"text": "a", "messages": []},
        {"messages": []},
        {"text": "a", "reset": "true"},
        {"text": "a", "unknown": True},
        {"messages": [{"role": "user", "content": 123}]},
    ],
)
def test_text_invalid_requests(assets: dict[str, Path], value: object) -> None:
    with pytest.raises(ValidationError):
        preprocessing.text(value, assets={"tokenizer": assets["qwen3"]}, options={})


@pytest.mark.parametrize(
    "tokenizer,chat", [("qwen3", False), ("whisper", True), ("pocket", True)]
)
def test_messages_need_chat_support(
    assets: dict[str, Path], tokenizer: str, chat: bool
) -> None:
    with pytest.raises(ValueError, match="messages require|accepts plain text"):
        tokenize(
            [{"role": "user", "content": "Hi"}], assets=assets[tokenizer], chat=chat
        )


@pytest.mark.parametrize(
    ("name", "prompt"),
    [
        ("normal", "  hello\n\tworld  "),
        ("quote", 'she said "Hello!"'),
        ("comma", "the answer is yes,"),
        ("ellipsis", "wait…"),
        ("question", "really?"),
        ("unknown", "café ☀."),
    ],
)
def test_pocket_normalization_and_sentencepiece(
    assets: dict[str, Path],
    expected: dict[str, list[int] | list[float] | str],
    name: str,
    prompt: str,
) -> None:
    inputs = preprocessing.pocket_tts(
        prompt, assets={"tokenizer": assets["pocket"]}, options={}
    )
    tokens = inputs["tokens"]
    assert isinstance(tokens, np.ndarray) and tokens.dtype == np.int32
    np.testing.assert_array_equal(tokens, expected[f"pocket_{name}"])
    decoded = decode_tokens(tokens, assets=assets["pocket"])
    assert decoded == expected[f"pocket_{name}_decoded"]
    if name != "unknown":
        np.testing.assert_array_equal(
            tokenize(decoded, assets=assets["pocket"]), tokens
        )


@pytest.mark.parametrize(
    "value", ["", " \n\t", {"messages": [{"role": "user", "content": "Hi"}]}]
)
def test_pocket_needs_nonempty_plain_text(
    assets: dict[str, Path], value: object
) -> None:
    with pytest.raises(ValueError, match="nonempty plain text"):
        preprocessing.pocket_tts(
            value, assets={"tokenizer": assets["pocket"]}, options={}
        )


@pytest.mark.parametrize("dtype", [np.int32, np.uint64])
def test_text_postprocessing(
    assets: dict[str, Path],
    expected: dict[str, list[int] | list[float] | str],
    dtype: type[np.int32 | np.uint64],
) -> None:
    tokens = np.asarray(expected["chat_default"], dtype=dtype)
    arguments = {"tokenizer": assets["qwen3"]}
    decoded = postprocessing.text({"tokens": tokens}, assets=arguments, options={})
    assert decoded == "user\nHello, world!\nassistant\n"
    assert (
        postprocessing.text(
            {"tokens": tokens, "truncated": np.asarray(False)},
            assets=arguments,
            options={"skip_special_tokens": False},
        )
        == expected["chat_default_decoded"]
    )
    with pytest.raises(ValueError, match="truncated"):
        postprocessing.text(
            {"tokens": tokens, "truncated": np.asarray(True)},
            assets=arguments,
            options={},
        )


@pytest.mark.parametrize(
    "tokens",
    [
        np.asarray([1.0]),
        np.asarray([[1]], dtype=np.int32),
        np.asarray(1),
        np.asarray([True]),
    ],
)
def test_decode_rejects_non_token_arrays(
    assets: dict[str, Path], tokens: NDArray[np.generic]
) -> None:
    with pytest.raises(ValueError, match="rank-one integer"):
        decode_tokens(tokens, assets=assets["qwen3"])


def test_whisper_prompt_does_not_change_text_tokenization(
    assets: dict[str, Path], expected: dict[str, list[int] | list[float] | str]
) -> None:
    tokens = tokenize("Hello, world!", assets=assets["whisper"])
    np.testing.assert_array_equal(tokens, expected["whisper_transcript"])
    assert (
        decode_tokens(tokens, assets=assets["whisper"])
        == expected["whisper_transcript_decoded"]
    )
    preprocessing.whisper(
        {
            "path": str(Path(__file__).parent / "fixtures" / "speech.flac"),
            "language": "fr",
            "task": "translate",
        },
        assets={"tokenizer": assets["whisper"], "features": assets["whisper"]},
        options={},
    )
    np.testing.assert_array_equal(
        tokenize("Hello, world!", assets=assets["whisper"]), tokens
    )


@pytest.mark.parametrize("tokenizer", ["qwen3", "pocket"])
def test_empty_generated_sequence(assets: dict[str, Path], tokenizer: str) -> None:
    assert (
        postprocessing.text(
            {"tokens": np.asarray([], dtype=np.int32)},
            assets={"tokenizer": assets[tokenizer]},
            options={},
        )
        == ""
    )
