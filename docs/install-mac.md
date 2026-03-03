# Install on macOS

## Requirements

- Apple Silicon (ARM64/NEON) or Intel (x86_64/AVX2)
- Python 3.12 or newer

## 1. Check your Python version

Run:

```bash
python3 --version
```

If you see `Python 3.12.x` (or newer), continue to step 3.
If not, install/upgrade Python in step 2.

## 2. Install or upgrade to Python 3.12+

### Option A (recommended): install Python with uv

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal, then install Python 3.12:

```bash
uv python install 3.12
uv run --python 3.12 python --version
```

### Option B: install Python with Homebrew

```bash
brew update
brew install python@3.12
python3 --version
```

If `python3 --version` still does not show 3.12+, restart the terminal and run it again.

## 3. Install Trillim with uv (recommended)

Create a project and add Trillim:

```bash
mkdir trillim-demo
cd trillim-demo
uv init
uv python pin 3.12
uv add trillim
```

If you want voice features (`--voice`), install the voice extra:

```bash
uv add "trillim[voice]"
```

## 4. Verify the install

```bash
uv run python --version
uv run trillim --help
uv run trillim list
```

## 5. Run Trillim

With `uv`, prefix commands with `uv run`:

```bash
uv run trillim pull Trillim/BitNet-TRNQ
uv run trillim chat Trillim/BitNet-TRNQ
uv run trillim serve Trillim/BitNet-TRNQ
```

## 6. pip alternative (venv)

If you prefer `pip`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
pip install --upgrade pip
pip install trillim
```

Then run:

```bash
trillim --help
trillim list
```
