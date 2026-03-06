# Python Components

Trillim's Python SDK is built from composable components. `LLM`, `Whisper`, and `TTS` can be used directly in your own async code, or composed into `Server(...)` when you want HTTP endpoints.

## Usage Model

- Import the component you need from `trillim`.
- Call `await component.start()` before using `component.engine` or `llm.harness`.
- Call `await component.stop()` when finished, ideally in a `finally` block.
- Use `router()` only when you are mounting the component into a FastAPI app.

## Using Components Standalone

Example: run the LLM directly in your own async code:

```python
import asyncio

from trillim import LLM


async def main():
    llm = LLM("~/.trillim/models/Trillim/BitNet-TRNQ")
    await llm.start()
    try:
        messages = [{"role": "user", "content": "Write a one-line haiku about CPUs."}]
        async for chunk in llm.harness.run(messages):
            print(chunk, end="", flush=True)
        print()
    finally:
        await llm.stop()


asyncio.run(main())
```

Example: use Whisper and TTS directly without exposing any HTTP endpoints:

```python
import asyncio
from pathlib import Path

from trillim import TTS, Whisper


async def main():
    whisper = Whisper(model_size="base.en")
    tts = TTS()
    await whisper.start()
    await tts.start()
    try:
        text = await whisper.engine.transcribe(Path("recording.wav").read_bytes())
        audio = await tts.engine.synthesize_full(text, voice="alba")
        Path("speech.wav").write_bytes(audio)
    finally:
        await whisper.stop()
        await tts.stop()


asyncio.run(main())
```

`Whisper` and `TTS` require the optional `voice` extra. See [Server](server.md#voice-optional-dependencies) for installation details.

## Composing a Server

Use the same components inside `Server(...)` when you want HTTP routes and an OpenAI-compatible API:

```python
from trillim import Server, LLM, Whisper, TTS

# Inference only
Server(LLM("~/.trillim/models/Trillim/BitNet-TRNQ")).run()

# Inference + voice pipeline
Server(
    LLM("~/.trillim/models/Trillim/BitNet-TRNQ"),
    Whisper(model_size="base.en"),
    TTS(),
).run()

# TTS only
Server(TTS()).run()
```

## LLM Component

```python
from trillim import LLM

LLM(
    model_dir="~/.trillim/models/Trillim/BitNet-TRNQ",
    adapter_dir=None,          # optional LoRA adapter path
    num_threads=0,             # 0 = auto-detect
    trust_remote_code=False,
    lora_quant=None,           # "none", "int8", "q4_0", etc.
    unembed_quant=None,        # "int8", "q4_0", etc.
    harness_name="default",    # "default" or "search"
)
```

After `await llm.start()`, use `llm.harness.run(messages, ...)` for the high-level chat/completions flow.

Use `llm.engine` when you want lower-level control over inference, for example:

- Call `llm.engine.generate(token_ids=..., ...)` directly for raw token-in, token-out generation.
- Use `llm.engine.tokenizer` to encode prompts, decode output, or apply a chat template yourself.
- Inspect `llm.engine.arch_config`, `llm.engine.default_params`, and `llm.engine.stop_tokens` when you need model/runtime details in your own orchestration layer.

In practice, `llm.harness` is the easier choice for normal chat-style use, while `llm.engine` is the lower-level path for custom prompting or custom generation loops.

`harness_name="search"` uses the default search provider (`ddgs`). To change provider on a running server, call `POST /v1/models/load` with `search_provider`.

## Whisper Component

```python
from trillim import Whisper

Whisper(
    model_size="base.en",   # Whisper model size
    compute_type="int8",
    cpu_threads=2,
)
```

After `await whisper.start()`, call `await whisper.engine.transcribe(audio_bytes, language=...)`.

## TTS Component

```python
from trillim import TTS

TTS(
    voices_dir="~/.trillim/voices",  # where custom voices are stored
)
```

After `await tts.start()`, use `tts.engine.list_voices()`, `await tts.engine.register_voice(...)`, `await tts.engine.synthesize_full(...)`, or `tts.engine.synthesize_stream(...)`.

## Custom Routes

Access the underlying FastAPI app to add custom routes:

```python
from trillim import Server, LLM

server = Server(LLM("~/.trillim/models/Trillim/BitNet-TRNQ"))
app = server.app

@app.get("/health")
async def health():
    return {"status": "ok"}

server.run(host="0.0.0.0", port=8000)
```
