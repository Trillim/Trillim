# Running compiled models

`trillim run PATH --threads 4` loads the compiled model directory and reads one JSON object per line from standard input. Each object must provide exactly the inputs declared by TDL. Each response contains the declared outputs, preserving their array shapes. Diagnostics go to standard error.

For an LLM whose inputs are `tokens` and `reset`:

```json
{"tokens":[9707,11,358],"reset":true}
{"tokens":[9707,11,358,1234],"reset":false}
```

The process retains caches between requests. The model's TDL defines reset and prefix-reuse behavior. Tokenization and output decoding belong to the caller.

```python
from trillim.runtime import Runtime

with Runtime("compiled/model", threads=4) as model:
    print(model.inputs)
    output = model.run({"tokens": [9707, 11, 358], "reset": True})
    tokens = output["tokens"]
```

`Runtime.inputs` and `Runtime.outputs` describe logical dtypes, shapes and maximum shapes. Python inputs may be numeric NumPy arrays or JSON-compatible numeric values. The runtime checks names, ranks, fixed dimensions, upper bounds and integer ranges before executing. Returned arrays own their data and survive subsequent runs or closing the model. BF16 and packed numerical outputs decode to FP32 arrays.

Runtime verifies the manifest and selects a private backend from `backends/scalar`, `backends/x86` or `backends/arm`. The ARM backend handles the compiler's `neon` target on ARM64. Each backend checks its device capabilities and uses the shared allocator, mappings, Python interpreter and persistent native thread pool in `backends/cpu`. Kernels run sequentially and can divide each invocation across the pool. The benchmark uses this same pool. Source installs build and cache the native helper on first use, using the kernel compiler hierarchy.

The backend prepares manifest arithmetic as Python functions and retains unchanged kernel parameter structs. Cache positions, tensor dimensions, pointers and footprint checks still update as execution proceeds.

`Runtime.cancel()` can be called from another thread. During execution, native SIGINT/SIGTERM handlers signal cancellation directly so a long kernel can stop before returning to Python. Previous handlers are restored after the run. Cancellation and execution failures invalidate the session; close it and create another. `close()` requests cancellation before waiting for execution to finish and releases resources. V1 permits one active model execution per process.
