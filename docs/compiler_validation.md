# Compiler validation

- All executable, weight, kernel ABI, and profile formats remain version 1.
- BitNet-2B generates `, I` from `Hello` through scalar and AVX2 artifacts.
- Qwen3-0.6B generates ` Answer!` through scalar and AVX2 artifacts; both tokens match an independent NumPy forward with RoPE, grouped attention, and KV caching.
- Qwen's relative logit error is about `7e-6`; multi-turn prefix reuse matches fresh execution bit for bit and runs two head calls instead of four.
- Qwen3.5 executes its recurrent and attention text backbone with empty vision inputs; `Hello` generates `,` with finite logits.
- PocketTTS produces 1,920 finite audio samples from `Hello.` using a one-frame bound; the truncation flag is expected.
- Ternary-Bonsai-1.7B generates `: I` with the checkpoint's YaRN settings and native FP16 weights; every unique weight span matches its source hash.
- Low-depth native profiling records three samples and mean, median, and variance at 1/8/64 context points; concrete tensor shapes and parameters accompany each measurement.
- Ctrl+C during benchmarking returned exit code 130 in about 0.5 seconds and left neither published nor unfinished model output.
- Ruff, strict mypy, native backend builds, and all 68 existing tests pass; no tests were added.

Final artifacts and reports are under `build/compiler/`:

| Artifact directory | Evidence |
| --- | --- |
| `bitnet_final` | Scalar generation and verified canonical weights |
| `bitnet_avx2_final` | AVX2 generation, selected fusions, and low-depth profiles |
| `qwen06b_final` | Scalar generation, independent reference, and prefix reuse |
| `qwen06b_avx2_final` | AVX2 generation and low-depth profiles |
| `qwen35_scalar_final_guards` | Text inference through recurrent state and KV caches |
| `pocket_final` | Audio generation and truncation metadata |
| `bonsai17_final` | FP16 generation, payload readback, and tied-weight aliasing |

Validation uses a build-only native execution adapter. Production runtime and backend sources are unchanged. Full vision conditioning, full-length TTS utterances, and checkpoints absent locally were not validated. Timing observations are functional validation data, not performance comparisons.
