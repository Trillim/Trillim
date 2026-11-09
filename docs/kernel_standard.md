# Kernel ABI 1

`trillim kernels` rebuilds all built-in kernels for the host backend and the scalar reference backend. `--backend scalar|x86|neon` selects one backend instead. The source build currently targets the host architecture; cross-platform release builds are separate.

`trillim kernels path/to/kernel.c --backend x86` compiles that kernel and relinks it with the installed objects. Multiple source paths are accepted. An existing kernel with the same exported name is replaced; a new name is added. No source paths means a full rebuild, removing previously added custom kernels from that backend.

Each backend is published at `$TRILLIM_HOME/1.0.0/kernels/<backend>/`, containing `kernels.so`, `backend.o`, `index.json` and an `objects/` directory. The index lists kernel symbols and has format version 1. Individual objects are retained for incremental linking; packages must include them to support updates without the original sources. Kernel discovery loads the library and index without invoking a C compiler or reading kernel sources.

The native benchmark helper uses the same compiler hierarchy and is cached separately under `bench/`. Benchmarking builds that helper if it is absent.

Compilers are tried in order: `zig cc`, an explicit `CC` command if set, then `cc`. On Windows, MinGW variants follow; `gcc` and `clang` are additional fallbacks. Unavailable tools are skipped. A failed compiler attempt is discarded before trying the next; compilation fails only after all available candidates fail. The current sources require a GCC-compatible driver, including Zig, GCC, Clang or MinGW.

Builds of one backend are serialized. Each build prepares a complete directory before atomically replacing the backend symlink. Existing resolved build directories remain available to active compiler and benchmark processes. Failure or cancellation removes the incomplete build and preserves the installed backend. All build files and Zig caches stay beneath `TRILLIM_HOME`, which defaults to `~/.trillim`.

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

Parameters contain fixed-width scalars, `size_t` values, and borrowed arrays with explicit lengths. Their native layout comes from the compiled kernel. There is no universal parameter struct or implicit argument order beyond each kernel's declared slots.

The requirements function reports all readable input bytes, writable output bytes, and scratch bytes for an invocation. Kernels must stay within those spans. Outputs and temporary workspace share the compiler's arena; live outputs cannot be overwritten. Kernels must not allocate heap memory or retain pointers after returning.

Only one kernel runs at a time. Its borrowed thread pool runs numbered tasks synchronously; workers claim tasks as they become free. Returning from the pool drains all tasks, including after failure or cancellation. Nested pool calls are unsupported.

Kernels check cancellation between bounded chunks of work and return `TRILLIM_CANCELLED`. Any other failure ends execution of the model. No output or cache state from a failed invocation may be reused.

Caches are persistent allocations with explicit capacity and update policy. A cache read preserves the values and dimensions observed at that point, even if a later instruction updates the cache. Direct cache writes or reads by a fused kernel must declare the same effects and respect capacity.
