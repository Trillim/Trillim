# Runtime validation and performance

The public `Runtime` API and `trillim run` execute compiler-generated manifests through one shared CPU backend. Scalar and x86 select different kernel libraries; allocation, cache handling, control flow, cancellation and the thread pool share an implementation. See [runtime usage](runtime.md).

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
