# What Is Trillim?

Trillim is the platform for everything local AI. DarkNet is the CPU inference engine powering Trillim.

## What Trillim Does

- Runs local language models on CPU, without requiring CUDA or ROCm.
- Supports ternary BitNet models and Llama-style architectures.
- Ships with a CLI, a Python SDK, and an OpenAI-compatible API server.
- Supports PEFT LoRA adapters on top of ternary base models.
- Optionally adds speech-to-text and text-to-speech through the `voice` extra.

## Why CPU Inference Matters

Most LLM stacks assume a GPU. BitNet models use ternary weights, so matrix multiplications reduce to additions and subtractions. That makes CPU inference practical on modern x86 and ARM systems when the runtime is optimized for the platform.

DarkNet is built around that constraint:

- AVX2 is used on x86_64 systems.
- NEON is used on ARM64 systems.
- Quantized runtime formats keep memory use lower and improve cache behavior.

## Good Fit

Trillim is a good fit when you want:

- local inference on a laptop, desktop, or CPU-only server
- a self-hosted API with OpenAI-compatible routes
- local experimentation with BitNet models and LoRA adapters
- a lower-cost inference path than a GPU deployment

## Model and Platform Support

| Category | Support |
|---|---|
| Architectures | `BitnetForCausalLM`, `LlamaForCausalLM` |
| CPU targets | x86_64 with AVX2, ARM64 with NEON |
| Threading | defaults to `num_cores - 2`, configurable with `--threads` |

## License

The Trillim Python SDK source code is MIT-licensed. The bundled C++ inference binaries (`inference`, `trillim-quantize`) are proprietary. You may use them as part of Trillim, but may not reverse-engineer or redistribute them separately.
