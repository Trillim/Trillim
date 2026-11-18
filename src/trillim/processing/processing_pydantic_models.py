from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class _ProcessingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    pre: Literal["text", "whisper", "parakeet", "pocket_tts", "image"] | None = None
    post: Literal["text", "audio", "image"] | None = None
    assets: dict[str, str] = Field(default_factory=dict)
    options: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_defaults(self) -> _ProcessingSpec:
        tokenizer = (
            self.pre in {"text", "whisper", "pocket_tts", "image"}
            or self.post == "text"
        )
        if tokenizer and "tokenizer" not in self.assets:
            raise ValueError("text processing requires tokenizer assets")
        if self.pre == "whisper" and "features" not in self.assets:
            raise ValueError("Whisper preprocessing requires feature-extractor assets")
        allowed = {"chat", "vision"} if self.pre == "text" else set()
        if self.post == "text":
            allowed.add("skip_special_tokens")
        if self.post == "audio":
            allowed.add("sample_rate")
            rate = self.options.get("sample_rate")
            if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
                raise ValueError("audio postprocessing requires a positive sample_rate")
        if set(self.options) - allowed:
            raise ValueError(
                f"unsupported processing options: {sorted(set(self.options) - allowed)}"
            )
        for name in ("chat", "vision", "skip_special_tokens"):
            if name in self.options and not isinstance(self.options[name], bool):
                raise ValueError(f"{name} must be boolean")
        return self


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: str
    content: str


class _TextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str | None = None
    messages: list[_Message] | None = None
    reset: bool = False

    @model_validator(mode="after")
    def _validate_content(self) -> _TextRequest:
        if (self.text is None) == (self.messages is None) or self.messages == []:
            raise ValueError("provide text or a nonempty messages list")
        return self


class _ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str
    height: int = Field(default=1024, ge=16, le=1920, multiple_of=16)
    width: int = Field(default=1024, ge=16, le=1920, multiple_of=16)
    seed: int = Field(default=0, ge=0, le=2**64 - 1)
    steps: int = Field(default=4, ge=1, le=1000)


class _AudioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    language: str = "en"
    task: Literal["transcribe", "translate"] = "transcribe"
