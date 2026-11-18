from __future__ import annotations

from functools import partial
from pathlib import Path
from threading import Event, RLock
from types import TracebackType
from typing import Self, cast

from ..compiler.artifacts.artifact_pydantic_models import _ExecutableManifest
from ..processing import Postprocessor, Preprocessor, postprocessing, preprocessing
from .errors import CancelledError
from .session import Runtime


class Model[Input, Output]:
    """Run preprocessing, a persistent Runtime, and postprocessing as one call.

    Supplied callables override the corresponding manifest defaults. Both
    processors must be available before the runtime allocates model resources.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        preprocess: Preprocessor[Input] | None = None,
        postprocess: Postprocessor[Output] | None = None,
        threads: int | None = None,
    ) -> None:
        if (
            preprocess is not None
            and not callable(preprocess)
            or postprocess is not None
            and not callable(postprocess)
        ):
            raise TypeError("processors must be callable functions")
        directory = Path(directory).expanduser().resolve()
        manifest = _ExecutableManifest.model_validate_json(
            (directory / "executable.json").read_bytes()
        )
        spec = manifest.processing
        if (preprocess is None and (spec is None or spec.pre is None)) or (
            postprocess is None and (spec is None or spec.post is None)
        ):
            raise ValueError(
                "Model requires both preprocessing and postprocessing functions, supplied explicitly or declared in TDL"
            )
        assets: dict[str, Path] = {}
        if spec is not None and (preprocess is None or postprocess is None):
            for name, relative in spec.assets.items():
                path = (directory / relative).resolve()
                if not path.is_relative_to(directory) or not path.is_dir():
                    raise ValueError(
                        f"missing or invalid processing asset directory: {relative}"
                    )
                assets[name] = path
        if preprocess is None:
            assert spec is not None and spec.pre is not None
            functions = {
                "text": preprocessing.text,
                "whisper": preprocessing.whisper,
                "parakeet": preprocessing.parakeet,
                "pocket_tts": preprocessing.pocket_tts,
                "image": preprocessing.image,
            }
            preprocess = cast(
                Preprocessor[Input],
                partial(functions[spec.pre], assets=assets, options=spec.options),
            )
        if postprocess is None:
            assert spec is not None and spec.post is not None
            processors = {
                "text": postprocessing.text,
                "audio": postprocessing.audio,
                "image": postprocessing.image,
            }
            postprocess = cast(
                Postprocessor[Output],
                partial(processors[spec.post], assets=assets, options=spec.options),
            )
        self._preprocess, self._postprocess = preprocess, postprocess
        self._lock = RLock()
        self._cancelled = Event()
        self._closed = False
        self._runtime = Runtime(directory, threads=threads)

    def run(self, value: Input) -> Output:
        """Process one request; conversation history must be supplied explicitly."""
        with self._lock:
            if self._closed:
                raise RuntimeError("model is closed")
            if self._cancelled.is_set():
                raise CancelledError("model execution cancelled")
            inputs = self._preprocess(value)
            if self._cancelled.is_set():
                raise CancelledError("model execution cancelled")
            outputs = self._runtime.run(inputs)
            if self._cancelled.is_set():
                raise CancelledError("model execution cancelled")
            result = self._postprocess(outputs)
            if self._cancelled.is_set():
                raise CancelledError("model execution cancelled")
            return result

    def cancel(self) -> None:
        self._cancelled.set()
        self._runtime.cancel()

    def close(self) -> None:
        self.cancel()
        with self._lock:
            if not self._closed:
                self._closed = True
                self._runtime.close()

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("model is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
