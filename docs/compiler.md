# Compiler

`trillim compile MODEL.tdl model=/path/to/model adapter=/path/to/adapter`

Kernel libraries must already be installed. From a source checkout, run `trillim kernels` to rebuild the host and scalar backends, or `trillim kernels path/to/kernel.c --backend x86` to update individual kernels. See [kernel_standard.md](kernel_standard.md) for the build layout and compiler fallback order. Model compilation does not build kernels implicitly.

The public Python modules are `trillim.tdl` (including `core`), `trillim.processing` and `trillim.runtime`. Compilation is exposed through the command above; compiler functions, types and helpers are private.

Generated files use `$TRILLIM_HOME/1.0.0/`, where `TRILLIM_HOME` defaults to `~/.trillim`. Only version 1.0.0 is read. `kernels/` holds compiled kernels and their symbol index, `backends/cpu/` holds the worker executable, and `compiled/` holds model bundles. Measurements are neither cached nor written to manifests or sidecar files. Benchmark helpers and toolchain scratch are temporary and removed after use.

The compiler detects the host backend and writes to `~/.trillim/1.0.0/compiled/<model_name>` by default. With one checkpoint, its directory name supplies `model_name`; for a safetensors file, the parent directory name does. With multiple checkpoints or no checkpoints, the TDL filename supplies the name.

- `--output PATH` chooses another destination.
- `--backend scalar|x86|neon` overrides host detection. Kernels declare their hardware features separately, such as AVX2, AVX512, VNNI, or NEON dotprod.
- `--depth none` favors graph coverage, native hardware specialization, and fixed-dtype implementations over generic kernels, without benchmarking. Software-library requirements do not increase specialization rank.
- `--depth low` refines that plan with fresh region measurements at single-row decode and 64-row prefill workloads, with a 120-second tuning budget. Comparisons include casts and other operations introduced by a split fusion.
- `--depth high` considers more internal fusion boundaries and adds 512-row prefill workloads, with a 300-second total tuning budget, including the low pass. Both tuned modes retain the incumbent unless a measured replacement improves mean relative cost by at least 3%, with no sampled workload more than 3% slower. Unknown or timed-out alternatives preserve the incumbent. Both emit one plan, avoiding whole-request choices that optimize prefill at the expense of every decode step.
- `--threads N` overrides the detected CPU thread count.
- `--max-input-size N` sets the default maximum length of variable input dimensions, defaulting to 4096; explicit TDL `maximum_shape` bounds take precedence. Fixed dimensions and input values are unaffected; cache capacities remain defined in TDL.

`Input` can declare scalar integer `minimum`, `maximum` and `multiple_of` constraints, plus per-input `maximum_shape` bounds for variable dimensions. Bounds guide allocation and remain runtime guards.

Each `NAME=PATH` supplies a TDL `Model("NAME")` declaration or a processing asset binding. Checkpoint paths must exist, be files or directories, and be accessible; these checks run during argument validation before tracing TDL or creating output.

The default depth is `none`. TDL and checkpoint tensors supply dtypes. The compiler computes scratch and cache requirements without a user-supplied memory budget. Benchmark preparation has a 4 GiB memory limit, and individual benchmark workers have a timeout. Numerical comparisons against scalar references use dtype-aware error budgets, including the allowed INT8 activation arithmetic of packed projections. Tuning inherits the process's BLAS settings; use the same settings when compiling and running a tuned model.

Tuning groups repeated regions whose available rewrites, native parameters, symbolic shapes and bounds, dtypes, packing, and tensor wiring agree. Shape comparison uses the equalities already proved by selected loop matches. A choice must pass fresh measurements and scalar validation on two instances when a group has multiple instances, using their actual weights. The selected rewrite then applies to the group. Only decisions are retained within that compilation; timings are discarded and nothing is reused across compilations. Distinct symbolic dependencies remain separate even when sampled dimensions happen to agree.

FP8 safetensors projections using `F8_E4M3`, `weight_scale_inv` tensors and 128-by-128 weight blocks import as `fp8_e4m3`. Compiled weights retain the original E4M3FN codes and their exact scale values, stored in an FP32 block-scale grid. Scalar and ARM Linear kernels consume these packed weights with FP16, BF16 or FP32 activations and FP32 accumulation; FP16/BF16 output casts can be fused. This is FP8 weight storage with floating activation arithmetic. Other FP8 encodings and block layouts are rejected.

The compiler edits the traced TDL graph directly. It validates checkpoint references, infers shapes and dtypes, simplifies computations, matches kernel TDL, selects a covering, schedules work, and assigns shared memory. Compilation publishes a complete directory with one atomic rename. An existing output directory is preserved; Ctrl+C removes the unfinished directory and stops active benchmark workers.

Each selected workload plan gets a greedy schedule before allocation. A fused kernel is one scheduling unit, retaining dependencies from every operation it covers. Ready work is ranked by output storage plus kernel scratch minus input storage whose last use it consumes. Views share storage in this estimate. Original graph order breaks ties; cache effects, guards and region boundaries remain dependencies. Allocation then determines exact lifetimes and permitted overwrites for that order. Buffer identities and constant offsets stay independent of execution order so differently scheduled workload plans share the same files and public bindings.

Kernel matching tries the original structure and a cheap normalization of views and Add/Mul chains. Low/high compilation depths also try broader mathematical normalization. These are temporary comparison forms; the compiler continues to use the traced TDL graph for execution.

The broader rules cover arithmetic reassociation, distribution, constants, coefficients, powers, roots, exp/log and trig identities, comparisons and selections, tensor views, matrix products, common reductions, shape expressions and proven indexing round trips. Other pure operations retain their operator and normalized arguments. Rules requiring positive values, nonzero divisors, compatible shapes or valid indices run only when those conditions are proved. Generic reducers retain their bodies and initial values; recognized reductions preserve empty-input behavior.

Casts are treated as mathematically lossless during comparison, as agreed for this compiler stage. Native binding still checks physical dtypes, packing, strides and buffer spans. Kernel inputs must refer to existing tensors; matching does not invent an unmaterialized intermediate. Cache effects and control flow stay outside normalization. Replacing a pure region covers its unused internal computations too, while external users and shape guards can prevent a match.

Normalization caches expressions and retains links to original tensors and operators. Each broad attempt allows 4096 expression constructions and 4096 matching visits, limits distribution to 32 terms and stops after eight rewrite rounds. Tensor traversal stops at a materialized value after 64 active levels. Reaching a limit leaves the earlier matching options available. This pass finds additional kernel choices; it does not hoist work across loops or change scheduling.

# Executable format 1

`executable.json` is a compact single-line JSON object. All paths are relative to its directory.

- `inputs` and `outputs` map public names to tensor buffers.
- Optional `processing` records standard pre/post recipe identifiers, bundled asset paths and recipe options; see [processing.md](processing.md).
- `buffers` describe dtype, shape, strides, capacity, and storage. Storage is an arena span, caller input, weight, constant, cache, or another tensor's view.
- `arena_size` covers intermediate outputs and temporary workspace; their live intervals determine reuse. `cache_size` covers persistent state.
- `caches` describe fixed capacities, initial values, update policies, and offsets. Append overflow is an error; sliding caches retain their newest values.
- `kernels` contain library paths, symbols, parameter layouts, alignment, and a required capability bitmask with readable feature names.
- `program` contains native calls, views, guards, branches, loops, reductions, and explicit cache accesses. Guards execute where their referenced values exist.
- `expressions` stores shared shape calculations once; `{"expr": "e0"}` refers to a table entry. Each workload plan has its own expression table.
- Optional `workload` identifies a variable input dimension. The format supports alternative `plans`, each with its own buffer layout and program; current compilation emits one plan. Measurements are not serialized.

Shapes and parameters use integer expressions. `scalar` reads a tensor using its declared dtype; `dimension` reads a tensor's captured actual shape; `cache_length` reads current cache metadata. Other expressions combine these values using arithmetic or broadcasting rules. Every actual dimension must fit its declared maximum. Non-finite metadata floats use `{"float": "-inf"}`, `{"float": "inf"}`, or `{"float": "nan"}`; tensor constants remain binary.

An `if` executes its chosen branch and its native result-copy calls. A `while` borrows its state for condition inputs and reuses state storage for body inputs and results when liveness and kernel alias permissions allow it. Dead arena inputs can initialize a loop without copying. Swapped values, changing shapes and live snapshots retain separate buffers where needed. Shared transfers are views; remaining transfers use native copy calls, with `shape_from` capturing the actual shape before a transfer.

A generic `reduce` iterates retained coordinates and reduced coordinates, initializes scalar accumulators, executes its pure reducer program, and stores each result. Its native transfer kernel handles element decoding and strided reads; mathematical reducers remain ordinary kernel calls. Common reductions use fused kernels instead.

A cache read captures its dimensions immediately. `snapshot: true` requires independent values; a direct cache binding is permitted only when the compiler proves those values remain unchanged for their lifetime. `metadata_only` cache writes commit a preceding direct kernel write without copying it again. Capacity guards precede direct append writes.

The bundle also contains `weights.bin`, `constants.bin`, and the selected backend's library. Weight encoding is described in [compiler_weights.md](compiler_weights.md), and native execution in [kernel_standard.md](kernel_standard.md).
