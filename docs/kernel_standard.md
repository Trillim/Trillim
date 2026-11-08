# Kernel ABI 1

Each backend builds its kernel sources into one `backend.o` and one loadable library under `$TRILLIM_HOME/1.0.0/kernels/<backend>/<build-hash>/`. `TRILLIM_HOME` defaults to `~/.trillim`. A backend has no dependency on another backend's library.

`common.h` is the sole shared kernel header. It contains the ABI, native dtype access and basic indexing utilities. Each kernel lives in one `.c` file with its own parameter structs, static helpers and four exported functions; it includes no other project files.

A kernel exports four functions:

- The execution function accepts output pointers, input pointers, scratch, its parameter struct, and a borrowed execution context, in that order.
- `_tdl()` returns Python TDL defining `pattern(...)` and a parameter `bind(context)` function; matching uses the computation rather than node names.
- `_parameters()` describes the native struct's field types and offsets, pointer counts, alignment, required buffer sizes, scratch size, and permitted aliases.
- `_capabilities()` declares ABI version 1, the backend, required hardware features, and required software libraries.

The compiler validates capabilities before invoking feature-dependent code. Metadata exports must be safe on the baseline CPU.

These four native exports define the kernel API. The Python helpers used by built-in parameter binders are compiler-private.

Logical tensors use ordinary mathematical shapes. Contiguous storage uses the last dimension as the fastest changing axis. Views may use permuted or zero strides; kernels receive the strides they need in their own parameter structs. Packed weights retain their logical dtype while the compiler selects their physical layout.

Parameters contain fixed-width scalars, `size_t` values, and borrowed arrays with explicit lengths. Their native layout comes from the compiled kernel. There is no universal parameter struct or implicit argument order beyond each kernel's declared slots.

The requirements function reports all readable input bytes, writable output bytes, and scratch bytes for an invocation. Kernels must stay within those spans. Outputs and temporary workspace share the compiler's arena; live outputs cannot be overwritten. Kernels must not allocate heap memory or retain pointers after returning.

Only one kernel runs at a time. Its borrowed thread pool runs numbered tasks synchronously; workers claim tasks as they become free. Returning from the pool drains all tasks, including after failure or cancellation. Nested pool calls are unsupported.

Kernels check cancellation between bounded chunks of work and return `TRILLIM_CANCELLED`. Any other failure ends execution of the model. No output or cache state from a failed invocation may be reused.

Caches are persistent allocations with explicit capacity and update policy. A cache read preserves the values and dimensions observed at that point, even if a later instruction updates the cache. Direct cache writes or reads by a fused kernel must declare the same effects and respect capacity.
