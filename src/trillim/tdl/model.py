from __future__ import annotations

from pydantic import ConfigDict, RootModel, field_validator

from .tdl_pydantic_models import Weight


class Model(RootModel[str]):
    model_config = ConfigDict(frozen=True)

    @field_validator("root")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.isalpha():
            raise ValueError("model names must contain only letters")

        return value

    @property
    def name(self) -> str:
        return self.root

    def __getitem__(self, tensor_name: str) -> Weight:
        return Weight(f"{self.name}.{tensor_name}")
