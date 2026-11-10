# Runtime validation and performance

The public `Runtime` API and `trillim run` execute compiler-generated manifests through one shared CPU backend. Scalar and x86 select different kernel libraries; allocation, cache handling, control flow, cancellation and the thread pool share an implementation. See [runtime usage](runtime.md).

## BitNet: 256-token prefill and 128 decode steps

The current BitNet path uses all 30 layers, BF16 activations and KV caches, ternary projections, and an INT8 output head. The model TDL explicitly selects these logical dtypes. The x86 backend uses AVX-VNNI when available and retains AVX2 implementations for machines without it. Scalar implementations provide numerical references.

Both implementations reserve 4,096 cache tokens and run with four threads on the same four physical cores, CPU IDs 0, 2, 4 and 5. Measurements use the same 256-token English prompt, followed by 128 generated decode steps; the first prediction comes from prefill, so each run returns 129 tokens. Backbone, vocabulary projection and greedy selection are included. Loading, tokenization and printing are excluded. The compiled timing includes the model's body instructions; session setup and loop bookkeeping are reflected in the separately recorded total wall time.

Models are warmed before measurement, AC power and turbo remain enabled, and timed runs are separated by at least 60 seconds. Each accepted run starts below 48°C, has no major page faults and records less than 0.1 CPU core of competing work. No BLAS is used. Preliminary runs with a 385-token cache are excluded from this comparison.

Accepted-run medians:

| Implementation | Accepted samples | Prefill tokens/s | Generation tokens/s |
| --- | ---: | ---: | ---: |
| DarkNet old-main | 3 | 136.96 | 27.32 |
| Compiler-generated BitNet | 4 | 152.50 | 27.17 |

Prefill is 11.3% faster; generation remains 0.5% behind. Median total wall time is 6.43 seconds, compared with DarkNet's 6.55 seconds. Both paths use INT8 head codes, but Trillim's logical `row_int8` dtype stores BF16 row scales while DarkNet stores FP32 scales; identical greedy continuations are not guaranteed.

Changes retained after measurement:

- Shape inference bounds the last-token slice to one row, so the output head no longer reserves space for a full prefill's vocabulary scores.
- Matching finds the captured weights needed to combine RMSNorm and activation quantization; emission removes unused compositions, repeated guards and unnecessary view instructions.
- A decode body now has 743 instructions, including 528 kernel calls and 92 guards, compared with 2,083 instructions, 654 calls and 1,055 guards in the first local build.
- BF16 normalization, attention, residual addition and MLP activation kernels avoid FP32 activation buffers; ternary kernels reuse INT8 activations across projections.
- The output head uses row-INT8 weights in blocks of 32 output rows, selected by the compiler without exposing physical packing in model TDL.
- Prepared calls reuse checked pointers, C parameter storage and unchanged arrays; metadata-only cache operations avoid constructing NumPy arrays.
- Cancellation polling uses bounded chunks instead of polling for every four projection outputs.

The final compiled allocation is 11.80 MiB of intermediate space and 300.02 MiB of caches at a maximum input size of 256. The compiler, manifest format and kernel ABI remain version 1. Experiments with four-output vector stores, split dot-product accumulators, smaller head register groups and vectorized exponentials were discarded when kernel or full-model measurements showed no reliable gain.

Validation covered numerical kernel references, AVX2/VNNI parity, packed versus canonical head results, partial tiles, argmax ties/NaNs, changing invocation sizes, session reuse and cancellation. Reusing a prefix produced the same 129 outputs as reset/full prefill. Decode cancellation completed in about 0.22 ms. Lint, strict mypy and all 68 existing tests passed; no tests were added.

The executable is `build/bitnet-opt/bitnet-bf16/executable.json`; weights and the selected kernel library are alongside it. Raw records are in `build/bitnet-opt/bench-samples.jsonl`, with accepted final samples in `compare-final*.log` and the DarkNet reference in `compare-pinned.log`. Generated files stay under the repository's ignored `build/` directory. The ignored model recipe is `TDL/bitnet_2b.tdl`.

Files changed for this work:

- `compiler/passes/`: inference, matching, packing selection, allocation, emission, verification and final program cleanup.
- `compiler/artifacts/`, `compiler/weights/` and `compiler/compile.py`: reusable arguments, physical packing and weight representation selection.
- `compiler/kernels/` and `compiler/bench/`: bindings, packing metadata, build flags and equivalent benchmark inputs.
- `kernels/x86/` and `kernels/scalar/`: the optimized kernels and scalar references.
- `backends/cpu/executor.py` and `cpu_dataclasses.py`: prepared calls and cache metadata handling.
- `TDL/bitnet_2b.tdl`, `docs/kernel_standard.md`, `docs/compiler_weights.md` and this report.

## Earlier runtime validation

Full BitNet and Qwen3-0.6B were compiled from `../../models/trillim/base/`. The x86 runs generated 512 outputs with all 30 BitNet layers and all 28 Qwen layers. Full scalar models also ran, generating one output each: 791 for BitNet and 2776 for Qwen, matching their first x86 predictions. Repeated x86 runs produced identical outputs within each implementation. BitNet answered "The capital of France is Paris." Qwen produced a coherent mathematical continuation. Tokenization was used only to inspect results, outside the runtime.

Measurements used an Intel i7-1255U, four threads, resident models, turbo enabled, and at least 60 seconds between timed runs. No BLAS was used. Every timed run started below 48°C. Background CPU use at the final runs' start was 0.08–0.55%. No compilation or diagnostics ran alongside the reported final samples. One overlapping attempt was discarded.

Each sample includes prefill, projection to vocabulary scores, greedy selection and 512 outputs: the first prediction followed by 511 decode steps. Prompts contained 14 BitNet tokens or three Qwen tokens. Model loading and printing are excluded. EOS stopping was disabled in the benchmark TDL. These are generation measurements, not a 512-token prefill matrix.

Two samples per model and implementation, using total outputs divided by total measured time:

| Model | DarkNet old-main | Initial measured runtime | Updated runtime | Updated sample times |
| --- | ---: | ---: | ---: | --- |
| BitNet | 26.93 tokens/s | 3.67 tokens/s | 8.91 tokens/s | 55.88 s, 59.00 s |
| Qwen3-0.6B | 34.63 tokens/s | 6.57 tokens/s | 8.47 tokens/s | 62.15 s, 58.74 s |

The earlier BitNet measurement used `65b192f`, Qwen used `cf1e1ed`, and the updated runtime used `1474523`. BitNet improved about 2.4× over its initial runtime measurement. The Qwen baseline already includes the BF16 projection fix; its original scalar-projection path was interrupted, so there is no measured speedup for that fix. DarkNet remains about 3.0× faster on BitNet and 4.1× faster on Qwen. Some Trillim samples recorded thermal-throttle counter increments; the DarkNet samples did not. The measurements describe this laptop's sustained behavior, rather than a fixed-frequency kernel comparison.

The arithmetic differs between implementations. Both compiled models retain BF16 output-head weights and use FP32 computation there; DarkNet uses INT8 head weights. Qwen's compiled projections retain BF16 weights, while DarkNet quantizes them to Q8 during loading. Consequently, these results compare the current inference paths, not the runtime ABI alone. Greedy outputs need not match between implementations.

Changes retained after validation:

- Extended the AVX2 linear kernel to accept BF16 activations and outputs while accumulating in FP32; Qwen now selects AVX2 at all 197 projection sites instead of falling back to scalar at 168 sites.
- Reused unchanged C parameter structs and prepared manifest arithmetic once instead of repeatedly decoding JSON arithmetic trees.
- Reused calculated values across consecutive read-only shape guards.
- Shared the existing benchmark thread pool with execution, including worker draining on cancellation; a shorter idle-spin experiment showed no clear benefit and was discarded.
- Bounded compiler canonicalization work so full models compile without the excessive traversal seen during initial validation.

The remaining cost has two substantial parts:

- Python still handles many small calls. An isolated eight-output diagnostic made 9,388 native calls for BitNet and 10,659 for Qwen. Native calls took 0.617 of 1.082 seconds and 0.510 of 1.069 seconds respectively. This diagnostic uses shorter cache histories than the 512-output runs; it identifies overhead, not sustained throughput. Folding more shape calculations in the compiler and preparing execution arguments more directly should reduce this cost.
- Numerical kernels still differ from DarkNet's specialized kernels. Quantized head/projection kernels and better fused coverage are separate improvements; silently changing a model's chosen logical dtype would not be an appropriate runtime optimization.

There is also excess planned memory. BitNet reserves 325.52 MiB of intermediate space plus 75 MiB of caches; Qwen reserves 408.77 MiB plus 112 MiB. The head output alone reserves 250.5 MiB and 296.75 MiB because its inferred upper bound permits 512 rows, although TDL selects the last row before the head. Tightening that bound is a compiler improvement. The backend currently allocates the manifest's declared space.

Validation also covered BF16 linear results against NumPy, changing-size packed outputs, cache/control-flow examples, input rejection, persistent sessions and cancellation. The scalar one-layer BitNet agreed with x86. A BF16 Qwen comparison selected different tokens at a near tie: the leading scalar scores differed by only 0.00023, with 0.0029 RMS difference across scalar/x86 vocabulary scores. FP32 versions generated the same tokens. This is consistent with the numerical contract, which does not promise identical rounding or greedy choices.

Lint, strict mypy and all 68 existing tests passed. No tests were added. The CPU helper also cross-compiled for Windows x86-64, Windows ARM64 and Linux ARM64; execution was validated on Linux x86-64. API cancellation stopped execution in under a millisecond in the diagnostic, while SIGINT/SIGTERM subprocess shutdown took approximately 81–142 ms.

Local raw records and diagnostics are under `build/runtime/`: `latest.jsonl`, `compare-bitnet.jsonl`, `compare-qwen-final.jsonl`, `native-short-isolated.jsonl`, `compare-logits.log`, `validate-fp32.log`, `validate-full-scalar.log`, `cancellation-final.log` and `cross-compile.log`. Executable manifests are `bitnet-full/executable.json` and `qwen-bf16/executable.json`. These generated files are intentionally untracked.

## Qwen INT8 and Whisper base, 2026-11-10

Qwen3-0.6B now matches DarkNet's performance with small measured gains. Whisper base takes about 13% less time than the official PyTorch model on the 9.984-second Ryan clip. These measurements use an Intel i7-1255U, four threads and CPU affinity 0,2,4,5. Each reported median contains two accepted runs, after warming the model and waiting at least 60 seconds between runs. Every accepted run starts below 48°C, stays on AC with turbo enabled, has no major page faults and records less than 0.1 CPU core of competing work.

| Qwen workload | DarkNet old-main | Trillim | Throughput change |
|---|---:|---:|---:|
| 256-token prefill | 217.70 tokens/s | 225.10 tokens/s | +3.40% |
| 128 decode steps | 35.24 tokens/s | 35.68 tokens/s | +1.25% |
| Complete run | 4.808 s | 4.767 s | 0.86% less time |

Both use the same prompt, all 28 layers, a 4,096-token cache, INT8 projections/head and BF16 activations/caches. Each run returns 129 predictions, including the prefill prediction. Trillim selects AVX-VNNI kernels on this CPU and also supplies AVX2 implementations. Neither implementation uses BLAS. The INT8 formats differ: Trillim stores group/head scales as BF16, while DarkNet stores FP32 scales. Their quantization rounding also differs, so these are not precision-identical models.

| Whisper base workload | PyTorch | Trillim |
|---|---:|---:|
| Ryan clip, 9.984 s of audio | 0.981 s | 0.855 s |

The Whisper reference uses OpenAI Whisper 20250625 with PyTorch 2.8.0 CPU, FP32 weights/activations, four threads and a plain greedy loop using the official model and KV-cache implementation. Its predictions were checked against `whisper.decode`. Both receive the same precomputed 80×3000 mel tensor and four-token English transcription prefix, use no token suppression, and produce 32 tokens including EOS. Loading, audio preprocessing, confidence scoring and text decoding are excluded. Trillim uses grouped INT8 projections, an INT8 head and BF16 activations/caches. Cross-attention K/V preparation happens before its decoding loop; total inference time includes that work on both sides.

### Implementation

- Added grouped INT8 scalar, AVX2 and VNNI projection kernels, using an eight-row/two-column prefill tile and bounded fallbacks for partial tiles.
- Shared Qwen's activation quantization across Q/K/V and gate/up projections, and added the `SiluMul` composition with independent input/output dtypes.
- Added fused Conv1D, LayerNorm, bias addition and GELU kernels for Whisper.
- Extended BF16 attention to 64-channel heads, shared K/V reads across four noncausal queries, and vectorized softmax.
- Shared RoPE angle calculation across heads and traversed its output in physical token order.
- Kept the existing compiler passes; compiler changes bind and validate parameters for the new kernels.

### Validation and limits

- Fixed-history Qwen comparisons cover both generated trajectories, each with 129 prediction points: mean KL divergence 0.0457–0.0849 nats, maximum 1.5225, and greedy agreement 114–118/129. These differences remain material despite valid kernel-level numerical checks; the measurements establish speed parity, not output-quality parity.
- Whisper's Ryan predictions exactly match PyTorch's 32 tokens. Mean KL divergence is 0.00764 nats, maximum 0.03766.
- Additional audio checks found only a punctuation difference on the first ten seconds of PG, and word differences on the Danish clip. No dataset-level accuracy claim is made.
- Reusing one Whisper session across Ryan, Danish, PG and Ryan again, with reset requested each time, reproduced the original Ryan output.
- Numerical checks cover grouped INT8 reference results, AVX2/VNNI parity, negative activation extremes, partial tiles, convolution boundaries, normalization, attention windows and mixed SiLU dtypes. GELU was checked over every BF16 bit pattern; its BF16 output differs from scalar at three finite inputs, by at most 1.53e-5. The softmax exponential sweep stayed within one FP32 ULP.
- Self-review fixed empty RoPE tensors accessing unallocated scratch and a potential extra BF16 read at the stride-two convolution boundary. Cancellation polling is bounded in the new loops. API cancellation checks returned in 0.066 ms for Qwen and 0.082 ms for Whisper.
- The review agent found no remaining P0/P1 issues. Lint, strict mypy and all 68 existing tests passed; no tests were added.

### Files and local evidence

- Compiler: `src/trillim/compiler/kernels/bindings.py`.
- TDL library: `src/trillim/tdl/activation.py` and `src/trillim/tdl/__init__.py`.
- Model recipes: `TDL/qwen3_06b.tdl` and new `TDL/whisper_base.tdl`; these remain ignored, following the existing repository convention.
- New scalar kernels: `src/trillim/kernels/scalar/group_int8_linear.c` and `gelu.c`.
- New x86 kernels: `src/trillim/kernels/x86/group_int8_avx2.c`, `group_int8_vnni.c`, `conv1d_avx2.c`, `layernorm_avx2.c`, `bias_add_avx2.c` and `gelu_avx2.c`.
- Updated x86 kernels: `attention_bf16_fma.c`, `rotary_avx2.c` and `silu_mul_avx2.c`.
- This report: `docs/runtime_performance.md`.

Fresh CLI-generated manifests and weights are in `build/qwen-int8/compiled/` and `build/whisper-bench/compiled/`. Their outputs agree with the benchmark artifacts. Timing records are `build/qwen-int8/bench-samples.jsonl` and `build/whisper-bench/bench-samples.jsonl`; accepted candidate records name the `fast8` artifacts. Numerical comparisons are in `build/qwen-int8/kl-final/results.json`, `build/whisper-bench/kl-results.json` and `build/whisper-bench/audio-validation.json`. Reference wrappers, checkpoint conversion and manual checks stay under `build/`.
