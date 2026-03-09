# CLI Reference

This page lists the main `trillim` subcommands and the flags most people use.

If you installed with `uv`, prefix each command on this page with `uv run`.

## `trillim list`

List models and adapters available on HuggingFace from the Trillim organization.

```bash
trillim list [--json]
```

| Flag | Description |
|---|---|
| `--json` | Output JSON instead of a formatted table |

Downloaded items are marked as local.

## `trillim pull`

Download a pre-quantized model from HuggingFace.

```bash
trillim pull <model_id> [--revision <ref>] [--force]
```

| Flag | Description |
|---|---|
| `model_id` | HuggingFace model ID such as `Trillim/BitNet-TRNQ` |
| `--revision` | Branch, tag, or commit hash to download |
| `--force`, `-f` | Re-download even if the model already exists locally |

Models are stored under `~/.trillim/models/<org>/<model>/`.

Example:

```bash
trillim pull Trillim/BitNet-TRNQ
```

## `trillim models`

List locally downloaded models and adapters.

```bash
trillim models [--json]
```

| Flag | Description |
|---|---|
| `--json` | Output JSON instead of a formatted table |

Example output:

```
Models
MODEL ID              ARCH        SIZE  SOURCE
--------------------  ----------  ----  -----
Trillim/BitNet-TRNQ   BitNet      1.2G  microsoft/bitnet-b1.58-2B-4T-bf16

Adapters
ADAPTER ID                        SIZE  COMPATIBLE MODELS
------------------------------    ----  -----------------
Trillim/BitNet-GenZ-LoRA-TRNQ      24M  Trillim/BitNet-TRNQ
```

## `trillim chat`

Start an interactive chat session with a model.

```bash
trillim chat <model_dir> [options]
```

| Flag | Description |
|---|---|
| `model_dir` | Local path or HuggingFace model ID resolved from `~/.trillim/models/` |
| `--lora <dir>` | Quantized LoRA adapter directory |
| `--threads <N>` | Inference thread count; `0` auto-detects as `num_cores - 2` |
| `--lora-quant <type>` | LoRA quantization: `none`, `bf16`, `int8`, `q4_0`, `q5_0`, `q6_k`, `q8_0` |
| `--unembed-quant <type>` | Unembed quantization: `int8`, `q4_0`, `q5_0`, `q6_k`, `q8_0` |
| `--trust-remote-code` | Allow loading custom tokenizer code from the model directory |
| `--harness <name>` | Harness name: `default` or `search` |
| `--search-provider <name>` | Search provider for the `search` harness: `ddgs` or `brave` |

Examples:

```bash
trillim chat Trillim/BitNet-TRNQ
trillim chat ./my-model-TRNQ
trillim chat Trillim/BitNet-TRNQ --lora Trillim/BitNet-GenZ-LoRA-TRNQ
trillim chat Trillim/BitNet-TRNQ --threads 4
trillim chat Trillim/BitNet-Search-TRNQ --harness search
trillim chat Trillim/BitNet-Search-TRNQ --harness search --search-provider brave
```

## `trillim serve`

Start an OpenAI-compatible API server.

```bash
trillim serve <model_dir> [options]
```

| Flag | Description |
|---|---|
| `model_dir` | Local path or HuggingFace model ID |
| `--host <addr>` | Bind address, default `127.0.0.1` |
| `--port <N>` | Bind port, default `8000` |
| `--voice` | Enable speech-to-text and text-to-speech endpoints |
| `--whisper-model <size>` | Whisper model size, default `base.en` |
| `--voices-dir <dir>` | Directory for persistent custom voice WAV files, default `~/.trillim/voices` |
| `--threads <N>` | Inference thread count; `0` auto-detects |
| `--lora-quant <type>` | LoRA quantization level |
| `--unembed-quant <type>` | Unembed quantization level |
| `--trust-remote-code` | Allow loading custom tokenizer code |

If you want `--voice`, install the optional extra first with `uv add "trillim[voice]"` or `pip install "trillim[voice]"`.

`trillim serve` starts with the default harness. To switch a running server to the search harness, call `POST /v1/models/load` with `"harness": "search"` and optional `"search_provider": "ddgs" | "brave"`.

Examples:

```bash
trillim serve Trillim/BitNet-TRNQ
trillim serve Trillim/BitNet-TRNQ --host 0.0.0.0 --port 3000
trillim serve Trillim/BitNet-TRNQ --voice
trillim serve Trillim/BitNet-TRNQ --voice --whisper-model medium.en
```

## `trillim quantize`

Quantize safetensors model weights and/or extract a LoRA adapter into Trillim's binary format. Only works for BitNet models currently.

```bash
trillim quantize <model_dir> [--model] [--adapter <dir>]
```

| Flag | Description |
|---|---|
| `model_dir` | HuggingFace model directory containing `config.json` and safetensors |
| `--model` | Write `<model_dir>-TRNQ/qmodel.tensors` and `rope.cache` |
| `--adapter <dir>` | Write `<adapter_dir>-TRNQ/qmodel.lora` |

You can pass both `--model` and `--adapter` in the same command.

Examples:

```bash
trillim quantize ./bitnet-2b --model
trillim quantize ./bitnet-2b --adapter ./my-lora-checkpoint
trillim quantize ./bitnet-2b --model --adapter ./my-lora-checkpoint
```
