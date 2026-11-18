# Processing and Model

`trillim.processing` is public. Its `preprocessing` and `postprocessing` modules contain standard model recipes; `utils` contains reusable tokenization, audio loading/resampling, mel extraction and image conversion functions. All model computation, including text encoders, vocoders and VAEs, stays in TDL.

Declare defaults at the top of a standalone `.tdl` file:

```python
from trillim.tdl import *

processing = {
    "pre": "text",
    "post": "text",
    "assets": {"tokenizer": "checkpoint"},
}
```

`checkpoint` refers to the CLI binding `checkpoint=/path/to/checkpoint`. The compiler copies tokenizer/configuration files and chat templates from that checkpoint into the compiled directory. It records the recipe identifiers, bundle-relative asset paths and options in `executable.json`. For a component checkpoint, conventional `tokenizer`/`feature_extractor` subdirectories or siblings are also searched. An explicit asset-only CLI binding can supply another directory. Processor Python code comes from the installed Trillim package. Loading a compiled model does not download assets or execute checkpoint-supplied Python.

Built-in preprocessing identifiers are `text`, `whisper`, `parakeet`, `pocket_tts` and `image`; postprocessing identifiers are `text`, `audio` and `image`. `text` uses local Hugging Face tokenizer assets and their chat template when available. SentencePiece-only assets are also supported. Whisper requires both `tokenizer` and `features` assets; both can reference the same checkpoint and are copied once. Standard text/audio postprocessors reject an explicit `truncated` output flag; raw Runtime callers can inspect partial outputs themselves. Audio output requires `options={"sample_rate": 24000}` for PocketTTS. Text options are `chat`, `skip_special_tokens`, and `vision` for text-only calls to the Qwen3.5/3.8 multimodal TDLs.

```python
from trillim.runtime import Model

with Model[str, str]("compiled/qwen") as model:
    print(model.run("Explain KV caching briefly."))
```

A `Model` resolves each explicitly supplied callback first, then the corresponding manifest default. Both preprocessing and postprocessing must be available; absent callbacks are an error, never an implicit raw-tensor mode. Caches persist across calls. For conversations, supply the complete history as `{"messages": [{"role": "user", "content": "..."}, ...]}`. Plain strings are independent prompts; the model TDL determines reusable prefixes. `{"text": "...", "reset": true}` requests a fresh session state.

For custom processing, omit `processing` or use `processing = {}` in TDL. Pass ordinary functions to `Model`:

```python
from pathlib import Path

import numpy as np

from trillim.processing import TensorInputs, TensorOutputs, utils
from trillim.runtime import Model

assets = Path("my-tokenizer")


def encode(prompt: str) -> TensorInputs:
    return {
        "tokens": utils.tokenize(prompt, assets=assets, chat=False),
        "reset": np.asarray(False, dtype=np.bool_),
    }


def decode(outputs: TensorOutputs) -> str:
    return utils.decode_tokens(outputs["tokens"], assets=assets)


with Model[str, str]("compiled/model", preprocess=encode, postprocess=decode) as model:
    print(model.run("Hello"))
```

The preprocessor receives exactly the object passed to `Model.run`. Return a mapping whose names, shapes and dtypes match the graph's `Input` declarations. The postprocessor receives a mapping of graph output names to independently owned NumPy arrays and can return any Python value. Validate application inputs inside your processor, using Pydantic where useful. Callbacks may be closures or callable objects to retain request metadata or loaded resources; Model serializes the whole call, including both processors.

To reuse a built-in recipe outside Model, bind its `assets` and `options` keyword arguments with `functools.partial`. To control the entire flow yourself:

```python
from trillim.runtime import Runtime

with Runtime("compiled/model") as runtime:
    result = decode(runtime.run(encode("Hello")))
```

`Model` and `Runtime` expose `run`, `cancel`, `close` and context-manager cleanup. Model checks cancellation between stages and forwards cancellation to Runtime. Custom callbacks run synchronously and must arrange their own cooperative cancellation if they perform long-running work. Cancellation invalidates the session; create a new instance afterward.

The CLI uses Model. It reads one plain-text or JSON request per line:

```sh
printf 'Hello\n' | trillim run compiled/qwen
printf '{"path":"speech.wav","language":"en"}\n' | trillim run compiled/whisper
printf 'Hello world.\n' | trillim run compiled/pocket --output speech.wav
printf '{"prompt":"A red kite","height":1024,"width":1024,"seed":7}\n' | trillim run compiled/flux --output kite.png
```

Text is printed directly. Audio postprocessing returns `Audio(samples, sample_rate)` and the CLI writes a WAV file. Image postprocessing returns a PIL image and the CLI saves it. Default output paths are `output.wav` and `output.png`; subsequent requests receive numbered filenames.

Whisper and Parakeet also accept an `Audio` object directly in Python. Whisper currently accepts one utterance of at most 30 seconds per call, with an explicit language and transcription/translation task. Parakeet uses its 128-bin, per-feature-normalized, unpadded mel recipe. PocketTTS uses its SentencePiece tokenizer with whitespace, capitalization and terminal-punctuation normalization. The Qwen3.5/3.8 text recipe supplies empty vision inputs; image/video requests require custom processing. Image generation uses the Klein prompt template, position IDs and flow-matching schedule; prompts are limited to 512 tokens. These recipes target the concrete architectures in `TDL`, not every checkpoint variant bearing the same family name.

Image height and width are bounded scalar inputs, each from 16 to 1920 inclusive and divisible by 16. The preprocessing request and compiled guards enforce those limits. `Input(..., minimum=16, maximum=1920, multiple_of=16)` supplies the compiler's value bounds; `maximum_shape` supplies per-input tensor dimension bounds when the global compile limit is unsuitable. The image programs reserve coordinate capacity for the largest latent grid plus the prompt, and derive all spatial shapes from runtime dimensions. Raw Runtime callers must also supply correctly sized coordinates and a sigma schedule ending at zero.
