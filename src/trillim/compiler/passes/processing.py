from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

from ...processing.processing_pydantic_models import _ProcessingSpec
from ...tdl.core.graph import Graph

_ASSETS = {
    "normalizer.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "tokenizer.vocab",
}


def _processing(
    graph: Graph, checkpoints: Mapping[str, Path], destination: Path
) -> _ProcessingSpec | None:
    if graph._processing is None:
        return None
    spec = _ProcessingSpec.model_validate(graph._processing)
    copied: dict[Path, str] = {}
    assets: dict[str, str] = {}
    for name, checkpoint in spec.assets.items():
        if not name.isidentifier():
            raise ValueError(f"invalid processor asset name: {name}")
        if checkpoint not in checkpoints:
            raise ValueError(f"missing processor checkpoint binding: {checkpoint}=PATH")
        source = checkpoints[checkpoint].resolve()
        source = source if source.is_dir() else source.parent
        required = (
            {
                "tokenizer.json",
                "tokenizer.model",
                "spiece.model",
                "sentencepiece.bpe.model",
                "vocab.json",
                "vocab.txt",
            }
            if name == "tokenizer"
            else {"preprocessor_config.json"}
            if name == "features"
            else _ASSETS
        )
        if not any((source / filename).is_file() for filename in required):
            folder = (
                "tokenizer"
                if name == "tokenizer"
                else "feature_extractor"
                if name == "features"
                else name
            )
            source = next(
                (
                    candidate
                    for candidate in (source / folder, source.parent / folder)
                    if candidate.is_dir()
                    and any((candidate / filename).is_file() for filename in required)
                ),
                source,
            )
        if not any((source / filename).is_file() for filename in required):
            raise ValueError(
                f"missing {name} assets in {source}; supply an asset checkpoint binding or leave processing empty for custom callbacks"
            )
        if source in copied:
            assets[name] = copied[source]
            continue
        relative = f"processing/{name}"
        target = destination / relative
        files = [
            path for path in source.iterdir() if path.is_file() and path.name in _ASSETS
        ]
        if not files:
            raise ValueError(
                f"no processor assets found for {checkpoint!r} in {source}"
            )
        target.mkdir(parents=True)
        for path in files:
            shutil.copyfile(path, target / path.name)
        templates = source / "chat_templates"
        if templates.is_dir():
            (target / "chat_templates").mkdir()
            for path in templates.glob("*.jinja"):
                shutil.copyfile(path, target / "chat_templates" / path.name)
        copied[source] = relative
        assets[name] = relative
    return spec.model_copy(update={"assets": assets})
