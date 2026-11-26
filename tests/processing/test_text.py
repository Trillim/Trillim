from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE

from trillim.compiler.artifacts.artifact_pydantic_models import _ExecutableManifest
from trillim.processing import (
    TensorInputs,
    TensorOutputs,
    postprocessing,
    preprocessing,
)
from trillim.processing.processing_pydantic_models import _ProcessingSpec
from trillim.processing.utils import _TextStream, _tokenizer, decode_tokens, tokenize
from trillim.runtime import Model, TokenEvent
from trillim.runtime import model as model_module


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


def test_streamed_text_waits_for_split_utf8(assets: dict[str, Path]) -> None:
    directory = assets["qwen3"]
    tokens = tokenize("Hello 🦄!", assets=directory, chat=False)
    eos = _tokenizer(directory).convert_tokens_to_ids("<|im_end|>")
    tokens = np.append(tokens, np.int32(eos))
    decoder = _TextStream(directory, skip_special_tokens=True)

    chunks = [decoder.push(int(token)) for token in tokens]
    text = decode_tokens(tokens, assets=directory)
    assert chunks == ["Hello", "", "", " 🦄", "!", ""]
    assert "".join(chunks) + decoder.finish(text) == text


@pytest.mark.parametrize("skip_special_tokens", [False, True])
def test_streamed_text_waits_for_byte_fallback_run(
    tmp_path: Path, skip_special_tokens: bool
) -> None:
    pieces = ["▁hello", "<0xC3>", "<0xA9>", "<eos>", "!", "?"]
    backend = Tokenizer(
        BPE(vocab={piece: index for index, piece in enumerate(pieces)}, merges=[])
    )
    backend.decoder = decoders.Sequence(
        [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(" ", 1, 0),
        ]
    )
    backend.add_special_tokens(["<eos>"])
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        '{"tokenizer_class": "PreTrainedTokenizerFast"}', encoding="utf-8"
    )
    tokens = np.asarray([0, 1, 2, 3, 1, 4, 1, 2, 5, 1], dtype=np.int32)
    decoder = _TextStream(tmp_path, skip_special_tokens=skip_special_tokens)

    chunks = [decoder.push(int(token)) for token in tokens]
    text = decode_tokens(
        tokens, assets=tmp_path, skip_special_tokens=skip_special_tokens
    )
    assert chunks[:3] == ["hello", "", ""]
    assert chunks[3] == ("" if skip_special_tokens else "é<eos>")
    assert chunks[5] == ("���!" if skip_special_tokens else "�!")
    assert chunks[8:] == ["é?", ""]
    tail = decoder.finish(text)
    assert tail == "�"
    assert "".join(chunks) + tail == text


def test_final_text_callback_blocks_reentry_and_closes_on_failure(
    assets: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    copyfile(assets["pocket"] / "tokenizer.model", tokenizer / "tokenizer.model")
    tokens = tokenize("Hello world", assets=tokenizer, chat=False)
    manifest = _ExecutableManifest(
        backend="scalar",
        threads=1,
        arena_size=0,
        cache_size=0,
        inputs={},
        outputs={},
        buffers={},
        caches={},
        kernels={},
        program=[],
        processing=_ProcessingSpec(post="text", assets={"tokenizer": "tokenizer"}),
    )
    (tmp_path / "executable.json").write_text(
        manifest.model_dump_json(), encoding="utf-8"
    )

    class ScriptedRuntime:
        def __init__(self) -> None:
            self.closed = False

        def run(
            self,
            inputs: TensorInputs,
            *,
            on_token: Callable[[TokenEvent], None] | None = None,
        ) -> TensorOutputs:
            if on_token is not None:
                for index, token in enumerate(tokens):
                    on_token(TokenEvent("tokens", index, int(token)))
            return {"tokens": tokens}

        def cancel(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    worker = ScriptedRuntime()
    monkeypatch.setattr(model_module, "Runtime", lambda directory, *, threads: worker)
    preprocessed: list[str] = []

    def preprocess(value: str) -> TensorInputs:
        preprocessed.append(value)
        return {}

    with Model[str, str](tmp_path, preprocess=preprocess) as model:

        def reenter(text: str) -> None:
            assert text == "Hello world"
            with pytest.raises(RuntimeError, match="model is already running"):
                model.run("nested")

        assert model.run("first", on_text=reenter) == "Hello world"
        assert preprocessed == ["first"]

        error = LookupError("text callback failed")

        def fail(text: str) -> None:
            raise error

        with pytest.raises(LookupError) as failure:
            model.run("second", on_text=fail)
        assert failure.value is error
        assert worker.closed
        with pytest.raises(RuntimeError, match="model is closed"):
            model.run("after failure")
        assert preprocessed == ["first", "second"]


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
