# Compiler

`trillim compile MODEL.tdl model=/path/to/model adapter=/path/to/adapter`

Kernel libraries must already be installed. From a source checkout, run `trillim kernels` to rebuild the host and scalar backends, or `trillim kernels path/to/kernel.c --backend x86` to update individual kernels. See [kernel_standard.md](kernel_standard.md) for the build layout and compiler fallback order. Model compilation does not build kernels implicitly.

The supported Python API is `trillim.tdl` and `trillim.tdl.core` for model authoring. Compilation is exposed through the command above; compiler functions, types and helpers are private.

Generated files use `$TRILLIM_HOME/1.0.0/`, where `TRILLIM_HOME` defaults to `~/.trillim`. Only version 1.0.0 is read. `kernels/` holds backend builds, `bench/` holds the native benchmark helper, and `compiled/` holds model bundles. Benchmark timings are reused only within the current compilation; the executable records the measurements used for selection.

The compiler detects the host backend and writes to `~/.trillim/1.0.0/compiled/<model_name>` by default. With one checkpoint, its directory name supplies `model_name`; for a safetensors file, the parent directory name does. With multiple checkpoints or no checkpoints, the TDL filename supplies the name.

- `--output PATH` chooses another destination.
- `--backend scalar|x86|neon` overrides host detection. Kernels declare their hardware features separately, such as AVX2, AVX512, VNNI, or NEON dotprod.
- `--depth none` picks a direct kernel plan with available backend specializations, without benchmarking.
- `--depth low` compares kernels for the operations in that plan at 1, 8, and 64 tokens.
- `--depth high` also compares fusion choices and extends workloads through 512, 4096, and 32768 tokens.
- `--threads N` overrides the detected CPU thread count.
- `--max-input-size N` sets the maximum length of each variable input dimension, defaulting to 4096. Fixed dimensions and input values are unaffected; cache capacities remain defined in TDL.
Each `NAME=PATH` must match a TDL `Model("NAME")` declaration. Checkpoint paths must exist, be files or directories, and be accessible; these checks run during argument validation before tracing TDL or creating output.

The default depth is `none`. TDL and checkpoint tensors supply dtypes. The compiler computes scratch and cache requirements without a user-supplied memory budget. Benchmark preparation has an internal memory limit, and individual benchmark workers have a timeout.

The compiler edits the traced TDL graph directly. It validates checkpoint references, infers shapes and dtypes, simplifies computations, matches kernel TDL, selects a covering, schedules work, and assigns shared memory. Compilation publishes a complete directory with one atomic rename. An existing output directory is preserved; Ctrl+C removes the unfinished directory and stops active benchmark workers.

Kernel matching first tries the original structure, then retries with a canonical view of pure expressions. It removes no-op casts and cancelling reshape/transpose chains from that view, and matches reassociated FP32 or integer Add/Mul chains. Associative searches are bounded to 16 terms and 256 assignments per candidate. Explicit dtype changes, cache effects and control flow remain boundaries. The executable graph is unchanged; candidate coverage, intermediate users, shape guards and native requirements retain their existing checks.

# Executable format 1

`executable.json` is a compact single-line JSON object. All paths are relative to its directory.

- `inputs` and `outputs` map public names to tensor buffers.
- `buffers` describe dtype, shape, strides, capacity, and storage. Storage is an arena span, caller input, weight, constant, cache, or another tensor's view.
- `arena_size` covers intermediate outputs and temporary workspace; their live intervals determine reuse. `cache_size` covers persistent state.
- `caches` describe fixed capacities, initial values, update policies, and offsets. Append overflow is an error; sliding caches retain their newest values.
- `kernels` contain library paths, symbols, parameter layouts, alignment, and a required capability bitmask with readable feature names.
- `program` contains native calls, views, guards, branches, loops, reductions, and explicit cache accesses. Guards execute where their referenced values exist.
- `expressions` stores shared shape calculations once; `{"expr": "e0"}` refers to a table entry. Each workload plan has its own expression table.
- `workloads` contain measured samples, mean, median, variance, reference comparisons, and reasons for skipped measurements. `plans` may provide different kernel coverings for workload ranges; each plan supplies its buffer layout and program.

Shapes and parameters use integer expressions. `scalar` reads a tensor using its declared dtype; `dimension` reads a tensor's captured actual shape; `cache_length` reads current cache metadata. Other expressions combine these values using arithmetic or broadcasting rules. Every actual dimension must fit its declared maximum. Non-finite metadata floats use `{"float": "-inf"}`, `{"float": "inf"}`, or `{"float": "nan"}`; tensor constants remain binary.

An `if` executes its chosen branch and its native result-copy calls. A `while` borrows its state for condition inputs and reuses state storage for body inputs and results when liveness and kernel alias permissions allow it. Dead arena inputs can initialize a loop without copying. Swapped values, changing shapes and live snapshots retain separate buffers where needed. Shared transfers are views; remaining transfers use native copy calls, with `shape_from` capturing the actual shape before a transfer.

A generic `reduce` iterates retained coordinates and reduced coordinates, initializes scalar accumulators, executes its pure reducer program, and stores each result. Its native transfer kernel handles element decoding and strided reads; mathematical reducers remain ordinary kernel calls. Common reductions use fused kernels instead.

A cache read captures its dimensions immediately. `snapshot: true` requires independent values; a direct cache binding is permitted only when the compiler proves those values remain unchanged for their lifetime. `metadata_only` cache writes commit a preceding direct kernel write without copying it again. Capacity guards precede direct append writes.

The bundle also contains `weights.bin`, `constants.bin`, and the selected backend's library. Weight encoding is described in [compiler_weights.md](compiler_weights.md), and native execution in [kernel_standard.md](kernel_standard.md).
