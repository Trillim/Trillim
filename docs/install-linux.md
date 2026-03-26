# Install on Linux

## Requirements

- Linux on x86_64 or ARM64
- Python 3.12 or newer
- `glibc 2.27` or newer

Check glibc:

```bash
ldd --version | head -n 1
```

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

Open a new shell, then install and verify Python 3.12:

```bash
uv python install 3.12
uv run --python 3.12 python --version
```

### Alternative: distro packages

Examples:

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip

# Fedora
sudo dnf install python3.12 python3.12-pip

# Arch Linux
sudo pacman -S python python-pip
```

Then verify:

```bash
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

For STT and TTS support:

```bash
uv add "trillim[voice]"
```

## 4. Verify the Install

```bash
uv run python --version
uv run trillim --help
```

## 5. First Run

```bash
uv run trillim models
uv run trillim pull Trillim/BitNet-TRNQ
uv run trillim chat Trillim/BitNet-TRNQ
```

Start the server:

```bash
uv run trillim serve Trillim/BitNet-TRNQ
```

## 6. pip Alternative

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

## Common Linux Pitfalls

- `trillim models` uses the network. `trillim list` is the local-only command.
- `trillim pull` only accepts repositories in the `Trillim/<name>` namespace.
- `trillim chat` and `trillim serve` load from managed store IDs only.
- If your distro Python is older than 3.12, use `uv` rather than forcing system packages.
