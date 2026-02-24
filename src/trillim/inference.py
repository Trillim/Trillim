# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
import asyncio
import os
import subprocess
import sys
import tempfile

from prompt_toolkit import prompt as better_input
from prompt_toolkit.key_binding import KeyBindings


def _make_key_bindings():
    """Create key bindings for the chat prompt. Ctrl+G opens $EDITOR."""
    kb = KeyBindings()

    @kb.add("c-g")
    def _open_editor(event):
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR", "vi")
        buf = event.app.current_buffer
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+", delete=False) as f:
            f.write(buf.text)
            tmp_path = f.name
        try:
            subprocess.call([editor, tmp_path])
            with open(tmp_path) as f:
                text = f.read()
            buf.text = text
            buf.cursor_position = len(text)
        finally:
            os.unlink(tmp_path)

    return kb


def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: trillim chat <model_directory> [--lora <adapter_dir>] "
            "[--threads N] [--lora-quant TYPE] [--unembed-quant TYPE] "
            "[--harness NAME]"
        )

    MODEL_PATH = sys.argv[1].strip()
    if len(MODEL_PATH) > 1 and MODEL_PATH[-1] == "/":
        MODEL_PATH = MODEL_PATH[:-1]

    # Parse --lora <adapter_dir>
    ADAPTER_DIR: str | None = None
    if "--lora" in sys.argv:
        lora_idx = sys.argv.index("--lora")
        if lora_idx + 1 < len(sys.argv) and not sys.argv[lora_idx + 1].startswith("--"):
            ADAPTER_DIR = sys.argv[lora_idx + 1]
        else:
            raise ValueError("--lora requires an adapter directory path.")

    TRUST_REMOTE_CODE = "--trust-remote-code" in sys.argv
    num_threads = 0
    if "--threads" in sys.argv:
        idx = sys.argv.index("--threads")
        if idx + 1 < len(sys.argv):
            num_threads = int(sys.argv[idx + 1])
    lora_quant = None
    if "--lora-quant" in sys.argv:
        idx = sys.argv.index("--lora-quant")
        if idx + 1 < len(sys.argv):
            lora_quant = sys.argv[idx + 1]
    unembed_quant = None
    if "--unembed-quant" in sys.argv:
        idx = sys.argv.index("--unembed-quant")
        if idx + 1 < len(sys.argv):
            unembed_quant = sys.argv[idx + 1]
    harness_name = "default"
    if "--harness" in sys.argv:
        idx = sys.argv.index("--harness")
        if idx + 1 < len(sys.argv):
            harness_name = sys.argv[idx + 1]

    config_path = os.path.join(MODEL_PATH, "config.json")

    try:
        from trillim.model_arch import ModelConfig as ArchConfig
        from trillim.utils import load_tokenizer, load_default_params
        from trillim.engine import InferenceEngine
        from trillim.harnesses import get_harness

        tokenizer = load_tokenizer(MODEL_PATH, adapter_dir=ADAPTER_DIR, trust_remote_code=TRUST_REMOTE_CODE)
        arch_config = ArchConfig.from_config_json(config_path, MODEL_PATH, adapter_dir=ADAPTER_DIR)
        stop_tokens = set(arch_config.eos_tokens)
        sampling_params = load_default_params(MODEL_PATH)

        engine = InferenceEngine(
            MODEL_PATH,
            tokenizer,
            stop_tokens,
            sampling_params,
            arch_config=arch_config,
            adapter_dir=ADAPTER_DIR,
            num_threads=num_threads,
            lora_quant=lora_quant,
            unembed_quant=unembed_quant,
        )
        harness_cls = get_harness(harness_name)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(engine.start())
            harness = harness_cls(engine)
            try:
                _run_chat_loop(loop, harness, sampling_params)
            finally:
                loop.run_until_complete(engine.stop())
        finally:
            loop.close()

    except BrokenPipeError:
        print("\nError: Inference engine crashed.")
        print("\nIf you think the engine is broken, please report the bug!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


def _run_chat_loop(loop, harness, sampling_params):
    """Interactive chat loop — sync input, async generation via harness."""
    model_name = os.path.basename(os.path.normpath(harness.engine.model_dir))
    max_context = harness.arch_config.max_position_embeddings
    messages = []

    kb = _make_key_bindings()
    print(f"Talk to {model_name} (Ctrl+D or 'q' to quit, '/new' for new conversation, Ctrl+G for editor)")
    while True:
        try:
            query = better_input("> ", key_bindings=kb)
        except (EOFError, KeyboardInterrupt):
            query = "q"

        if query.strip() == "q":
            break

        if query.strip() == "/new":
            messages = []
            harness.engine._cached_prompt_str = ""
            harness.engine.cached_token_ids = []
            print("Starting new conversation.")
            continue

        messages.append({"role": "user", "content": query})

        # Context limit check
        token_ids, _ = harness._prepare_tokens(messages)
        if len(token_ids) >= max_context:
            print(
                f"Context window full ({max_context} tokens). Starting new conversation."
            )
            messages = [messages[-1]]
            harness.engine._cached_prompt_str = ""
            harness.engine.cached_token_ids = []

        print("Model Response: ", end="", flush=True)
        loop.run_until_complete(_stream_response(harness, messages, sampling_params))
        print()


async def _stream_response(harness, messages, sampling_params):
    """Drain the harness async generator, printing chunks as they arrive."""
    async for chunk in harness.run(messages, **sampling_params):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    main()
