# Install on macOS

## Requirements

- macOS on x86_64 or Apple Silicon
- Python 3.12 or newer

`uv` is the recommended way to install both Python and Trillim.

## 1. Check Python

```bash
python3 --version
```

If that already prints `Python 3.12.x` or newer, continue to step 3.

## 2. Install Python 3.12+

### Recommended: install Python with `uv`

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal, then install and verify Python 3.12:

```bash
uv python install 3.12
uv run --python 3.12 python --version
```

### Alternative: Homebrew

```bash
brew update
brew install python@3.12
python3 --version
```

## 3. Install Trillim

Create a project and add the package:

```bash
mkdir trillim-demo
cd trillim-demo
uv init
uv python pin 3.12
uv add trillim
```

If you need speech-to-text or text-to-speech, install the extra:

```bash
uv add "trillim[voice]"
```

## 4. Verify the Install

```bash
uv run python --version
uv run trillim --help
```

## 5. First Run

List published bundles, pull one, and open a chat:

```bash
uv run trillim models
uv run trillim pull Trillim/BitNet-TRNQ
uv run trillim chat Trillim/BitNet-TRNQ
```

Start the local API server:

```bash
uv run trillim serve Trillim/BitNet-TRNQ
```

## 6. pip Alternative

If you prefer `pip`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
pip install trillim
```

With voice support:

```bash
pip install "trillim[voice]"
```

Then run:

```bash
trillim --help
```

## Common macOS Pitfalls

- If you installed with `uv`, remember the `uv run` prefix.
- `trillim chat` and `trillim serve` expect store IDs like `Trillim/BitNet-TRNQ`, not raw model paths.
- Voice features are not included unless you install `trillim[voice]`.
