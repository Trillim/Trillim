from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

import numpy as np
from numpy.typing import NDArray


class Device(Protocol):
    type: str


class Tensor(Protocol):
    dtype: object
    device: Device

    def __getitem__(self, key: object) -> Tensor: ...

    def __mul__(self, other: Tensor) -> Tensor: ...

    def __add__(self, other: Tensor) -> Tensor: ...

    def unsqueeze(self, dim: int) -> Tensor: ...

    def squeeze(self, dim: int) -> Tensor: ...

    def transpose(self, dim0: int, dim1: int) -> Tensor: ...

    def detach(self) -> Tensor: ...

    def float(self) -> Tensor: ...

    def cpu(self) -> Tensor: ...

    def numpy(self) -> NDArray[np.float32]: ...


class Module(Protocol):
    def eval(self) -> Self: ...

    def float(self) -> Self: ...

    def cpu(self) -> Self: ...

    def named_parameters(self) -> Iterator[tuple[str, Tensor]]: ...

    def load_state_dict(
        self, state: Mapping[str, Tensor], *, strict: bool
    ) -> object: ...

    def __call__(self, *args: object, **kwargs: object) -> object: ...


class Factory(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> Module: ...


class PretrainedFactory(Protocol):
    def from_pretrained(self, checkpoint: Path, **kwargs: object) -> Module: ...


class EncoderModel(Module, Protocol):
    encoder: Module


class MimiModel(Module, Protocol):
    quantizer: Module

    def decode_from_latent(self, projected: Tensor, state: State) -> Tensor: ...


class TorchAPI(Protocol):
    __version__: str
    float32: object

    def from_numpy(self, value: NDArray[np.float32]) -> Tensor: ...

    def set_num_threads(self, count: int) -> None: ...

    def set_num_interop_threads(self, count: int) -> None: ...

    def get_num_threads(self) -> int: ...

    def get_num_interop_threads(self) -> int: ...

    def set_default_device(self, device: str) -> None: ...

    def set_float32_matmul_precision(self, precision: str) -> None: ...

    def inference_mode(self) -> AbstractContextManager[object]: ...


class Archive(Protocol):
    def keys(self) -> list[str]: ...

    def get_tensor(self, name: str) -> Tensor: ...


class SafeOpen(Protocol):
    def __call__(
        self, filename: str, *, framework: str, device: str
    ) -> AbstractContextManager[Archive]: ...


type State = dict[str, dict[str, Tensor]]


class InitializeStates(Protocol):
    def __call__(
        self, model: Module, *, batch_size: int, sequence_length: int
    ) -> State: ...


class IncrementSteps(Protocol):
    def __call__(self, model: Module, state: State, *, increment: int) -> None: ...


class EncoderOutput(Protocol):
    last_hidden_state: Tensor


type Arrays = dict[str, NDArray[np.float32]]


@dataclass(slots=True)
class Adapter:
    forward: Callable[[], Arrays]
    reset: Callable[[], None] | None
    decode: Callable[[int], Arrays] | None
    modules: tuple[Module, ...]
    source_modules: tuple[str, ...]
    implementation: str
    scope: str
    exclusions: str
    reset_scope: str
    native_tensor_count: int | None = None
