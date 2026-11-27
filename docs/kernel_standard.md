# Kernel ABI 1

`trillim kernels` rebuilds all built-in kernels for the host backend and the scalar reference backend. `--backend scalar|x86|neon` selects one backend instead. The source build currently targets the host architecture; cross-platform release builds are separate.

`trillim kernels path/to/kernel.c --backend x86` compiles that kernel and relinks it with the installed objects. Multiple source paths are accepted. An existing kernel with the same exported name is replaced; a new name is added. No source paths means a full rebuild, removing previously added custom kernels from that backend.

Each backend is published at `$TRILLIM_HOME/1.0.0/kernels/<backend>/`, containing `kernels.so`, `index.json` and an `objects/` directory. The index lists kernel symbols and has format version 1. Individual objects are retained for incremental linking; packages must include them to support updates without the original sources. Kernel discovery loads the library and index without invoking a C compiler or reading kernel sources.

The runtime worker and benchmark use the same CPU thread-pool implementation. The worker executable is built separately; compilation builds a temporary benchmark helper and removes it after tuning. Neither benchmark measurements nor helper builds are cached.

Windows benchmark builds statically link Winpthreads and use CRT aligned allocation and QueryPerformanceCounter timing. Zig supplies Winpthreads; other Windows toolchains need its headers and static library. Windows kernels resolve standard mathematical functions through their CRT, so capability detection does not require a separate libm DLL.

Compilers are tried in order: Zig, GCC, Clang, then host MinGW on Windows. An explicit `CC` or `CXX` selects that compiler instead. Unavailable tools are skipped. A failed compiler attempt is discarded before trying the next; compilation fails only after all available candidates fail. The current sources require a GCC-compatible driver, including Zig, GCC, Clang or MinGW.

Kernels compile with `-O3` for the architecture's baseline CPU, including `-mcpu=baseline` with Zig. Optional instructions belong in functions with explicit target attributes and corresponding capability bits; building on a newer CPU must not silently add undeclared instruction requirements.

Builds of one backend are serialized. Each build prepares a complete directory before atomically replacing the backend symlink. Existing resolved build directories remain available to active compiler and benchmark processes. Failure or cancellation removes the incomplete build and preserves the installed backend. Toolchain scratch, including Zig caches, is confined to temporary build directories and removed after use.

A backend has no dependency on another backend's library. Portable scalar kernels are compiled into each backend's own library.

`common.h` is the sole shared kernel header. It contains the ABI, native dtype access and basic indexing utilities. Each kernel lives in one `.c` file with its own parameter structs, static helpers and four `TRILLIM_KERNEL` functions; it includes no other project files. The export marker keeps these entrypoints visible when building libraries with hidden visibility. The builder indexes the compiler-generated assembly, so C macros and conditional compilation are resolved by the C compiler.

A kernel exports four functions:

- The execution function accepts output pointers, input pointers, scratch, its parameter struct, and a borrowed execution context, in that order.
- `_tdl()` returns Python TDL defining `pattern(...)` and a parameter `bind(context)` function; matching uses the computation rather than node names.
- `_parameters()` describes the native struct's field types and offsets, pointer counts, alignment, required buffer sizes, scratch size, and permitted aliases.
- `_capabilities()` returns ABI version 1 and one `uint64_t` mask of required capabilities. The library's location identifies the backend.

`common.h` defines the capability flags. Bits 0–31 describe backend-specific hardware features; bits 32–63 describe software libraries. X86 flags include AVX2, AVX512 variants, AVX-VNNI, AVX512-VNNI and FMA. NEON flags cover NEON and dotprod. LIBM and BLAS use the same software bits across backends, so portable kernels retain their requirements when included in another backend.

The compiler builds one available-capabilities mask per backend and accepts a kernel when `(required & available) == required`. Missing or unknown bits make a kernel unavailable. Metadata exports must be safe on the baseline CPU. The executable records the required mask and readable feature names.

These four native exports define the kernel API. The Python helpers used by built-in parameter binders are compiler-private.

Logical tensors use ordinary mathematical shapes. Contiguous storage uses the last dimension as the fastest changing axis. Views may use permuted or zero strides; kernels receive the strides they need in their own parameter structs. Packed weights retain their logical dtype while the compiler selects their physical layout.

A kernel's TDL may declare `packings = {"weight": "row_int8_block32"}` for a checkpoint input. Undeclared inputs require `canonical` packing. Matching validates the logical computation and the binder's shape/stride restrictions; selection chooses the representation before weight conversion. The executable records each input's required packing, which verification checks against the weight manifest. A weight also needed by a canonical consumer currently selects canonical kernels throughout, rather than duplicating its storage. Packing names are compiler-private.

Attention and rotary tensors use `[tokens, heads, channels]`. KV caches use `[capacity, kv_heads, channels]` and append along axis 0. Splitting and merging heads are reshape views. GQA groups query heads without replicating K/V. The planner can bind a producer's contiguous output through reshape views directly to the next cache slot, with a capacity check before the producer runs.

`QuantizeInt8(x)` returns contiguous INT8 activation codes and an FP32 multiplier per row, shaped `[..., 1]`. The multiplier is `127 / max(abs(x), 1e-5)` along the final axis; codes round to nearest even. `QuantizedLinear(codes, weight, multiplier, dtype=...)` projects these shared codes and divides by the multiplier. Scalar and x86 kernels support ternary and row-INT8 weights, with FP16, BF16 or FP32 outputs. A model can quantize once for Q/K/V or gate/up instead of repeating the work in each projection. `BitLinear` composes both stages when sharing is unnecessary. Fused RMSNorm/quantization preserves the normalization output's dtype rounding before quantizing it. AVX-VNNI variants require both AVX2 and AVX-VNNI capability bits.

Parameters contain fixed-width scalars, `size_t` values, and borrowed arrays with explicit lengths. Their native layout comes from the compiled kernel. There is no universal parameter struct or implicit argument order beyond each kernel's declared slots.

The compiler emits an `argument_key` containing the changing source values that determine a call's parameters and buffer metadata. Within one input binding, the backend can reuse checked pointers and packed parameters when that key is unchanged; it still executes the kernel every time. Changed keys update the parameters, and rebinding inputs clears prepared calls. Unchanged parameter arrays retain their storage. This does not change the native ABI.

The requirements function reports all readable input bytes, writable output bytes, and scratch bytes for an invocation. Kernels must stay within those spans. Outputs and temporary workspace share the compiler's arena; live outputs cannot be overwritten. Kernels must not allocate heap memory or retain pointers after returning.

`can_alias(parameters, output_slot, input_slot)` permits an output pointer to equal the specified input pointer. A null callback forbids overlap. Every overlapping input slot must grant permission, including when multiple inputs share one tensor. The compiler first proves matching native dtype, shape and contiguous access. For dynamic parameters it passes `NULL`; the callback may grant permission only if it is safe for every legal invocation under those conditions. Otherwise it returns false. Permission must account for worker access order. Reused outputs remain separate logical tensors and appear as aliases in the manifest. Caller inputs, weights, caches and live snapshots are preserved.

Native `Scatter` can reuse its contiguous source buffer when indices and update values do not overlap it. With equal input/output pointers, it writes only the updated slices. Recurrent compositions use this to fill one output row per token.

Only one kernel runs at a time. Its borrowed thread pool runs numbered tasks synchronously; workers claim tasks as they become free. Returning from the pool drains all tasks, including after failure or cancellation. Nested pool calls are unsupported.

Kernels check cancellation between bounded chunks of work and return `TRILLIM_CANCELLED`. Any other failure ends execution of the model. No output or cache state from a failed invocation may be reused.

Caches are persistent allocations with explicit capacity and update policy. A cache read preserves the values and dimensions observed at that point, even if a later instruction updates the cache. Direct cache writes or reads by a fused kernel must declare the same effects and respect capacity.

Pure views whose metadata depends only on dimensions can be described by buffer aliases without an execution instruction. Views that capture scalar values or cache lengths, and control-flow bindings, retain explicit instructions. Metadata-only cache writes update lengths without constructing tensor arrays or copying values.
