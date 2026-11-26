# Running compiled models

`trillim run PATH --threads 4` reads one plain-text or JSON request per line. The Python `Model` API applies the processors declared in the executable manifest. Text is printed as stable decoded chunks arrive; audio and image outputs are saved with `--output PATH`. See [processing.md](processing.md) for processor declarations and custom callbacks.

```python
from trillim.runtime import Model

with Model[str, str]("compiled/model", threads=4) as model:
    text = model.run(
        "Explain why the sky is blue.",
        on_text=lambda chunk: print(chunk, end="", flush=True),
    )
    print()
```

`Model.run` still returns the complete processed result. `on_text` requires the built-in text postprocessor. Incremental decoding retains incomplete Unicode bytes until they form stable text. Tokenizers with model-specific decoding that can revise earlier text deliver their final text as one chunk.

For named tensors and raw token events, use `Runtime`:

```python
from trillim.runtime import Runtime

with Runtime("compiled/model", threads=4) as runtime:
    result = runtime.run(
        {"tokens": [9707, 11, 358], "reset": True},
        on_token=lambda event: print(event.index, event.token, flush=True),
    )
```

The example assumes a model with those inputs and a rank-one integer `tokens` output. `TokenEvent` contains `output`, `index` and `token`. Events follow the compiled model's output, including EOS when the model returns it. Callbacks run on the calling thread and should return promptly. Callback exceptions abort the request and close its worker. Reentering the same session during a callback is rejected.

## CPU execution

Python validates the manifest and inputs and owns session and processing APIs. Each session starts one persistent C++ CPU worker. It maps the existing weights and constants, allocates memory, maintains caches, evaluates dynamic shapes and control flow, and calls the existing C kernel ABI through the native thread pool. Scalar, ARM NEON and x86 executables use this shared executor, with their own kernel libraries and capability checks.

The C++ worker handles generation, EOS conditions and iteration limits from the manifest. Proven append-only token loops emit committed tokens immediately. Other graphs emit their final token output when it becomes available. A bounded pipe provides backpressure; generation does not require a Python acknowledgement for every token. Named output arrays are copied into independently owned NumPy arrays. BF16 and packed numerical outputs decode to FP32.

Build the CPU worker explicitly with `uv run task build-cpu`, or `uv tool run --from taskipy task build-cpu` without installing the project's dependencies. The recipe lives in `pyproject.toml` and calls `scripts/build_cpu.py`, using a C++20 compiler in this default order: Zig, GCC, Clang, then host MinGW on Windows. `CXX` explicitly overrides that selection. Kernel builds use the same order with a `CC` override, and compiler benchmarks use the C++ selection. It creates `~/.trillim/1.0.0/backends/cpu/worker`, or `worker.exe` on Windows. `TRILLIM_HOME` changes this installation root. Runtime launches that executable without compiling or caching backend code. Set `TRILLIM_CPU_WORKER` to use a prebuilt executable at another location. The worker links its thread pool and JSON parser directly; only model kernel libraries are loaded dynamically. Distribution of the executable is separate from runtime execution.

## State and cancellation

Caches persist between requests; model TDL defines reset, prefix reuse and sliding behavior. `Runtime.inputs` and `Runtime.outputs` expose logical dtypes and bounded shapes. Inputs may be numeric NumPy arrays or JSON-compatible values. The runtime validates input names, ranks, fixed dimensions, bounds and integer ranges.

`Runtime.cancel()` can be called from another thread. The worker is asked to stop and is killed and reaped if it does not exit within the cancellation grace period. Sessions opened on the main Python thread install SIGINT and SIGTERM forwarding for active inference, including inference on another thread. While idle, signals delegate to the application handlers. Closing the last session on the main thread restores those handlers. Explicit cancellation works from any thread. The worker also monitors its parent's pipe so an abrupt parent exit does not leave inference running. Cancellation and execution failures invalidate the session; close it and create another. Use context managers to close workers deterministically. V1 permits one active inference per Python process.

`tests/runtime` covers scalar and host execution, Qwen decoder cache reuse, packed caches and outputs, dynamic control flow, integer precision, streaming before completion, malformed protocol messages, parent/worker signals, callback failures and process cleanup. The full-width Qwen layer fixture uses original weights bundled under `tests/runtime/fixtures`.
