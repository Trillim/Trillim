# Install on Windows

## Requirements

- Windows on x86_64 or ARM64
- Python 3.12 or newer

`uv` is the recommended installer.

## 1. Check Python

In PowerShell:

```powershell
py --version
```

If that already prints `Python 3.12.x` or newer, continue to step 3.

## 2. Install Python 3.12+

### Recommended: `winget`

```powershell
winget install -e --id Python.Python.3.12
py -3.12 --version
```

### Alternative: python.org

Download Python 3.12+ from [python.org](https://www.python.org/downloads/windows/) and enable:

- `Add python.exe to PATH`
- `Install launcher for all users`

Then verify:

```powershell
py -3.12 --version
```

## 3. Install Trillim

Install `uv`:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new PowerShell window, then create a project and add Trillim:

```powershell
mkdir trillim-demo
cd trillim-demo
uv init
uv python pin 3.12
uv add trillim
```

For STT and TTS:

```powershell
uv add "trillim[voice]"
```

## 4. Verify the Install

```powershell
uv run python --version
uv run trillim --help
```

## 5. First Run

```powershell
uv run trillim models
uv run trillim pull Trillim/BitNet-TRNQ
uv run trillim chat Trillim/BitNet-TRNQ
```

Start the server:

```powershell
uv run trillim serve Trillim/BitNet-TRNQ
```

## 6. pip Alternative

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
pip install trillim
```

With voice support:

```powershell
pip install "trillim[voice]"
```

## Common Windows Pitfalls

- If activation is blocked, keep using `uv run ...`; you do not need an activated venv to use Trillim with `uv`.
- `trillim serve` binds to `127.0.0.1:8000` in the CLI. Use the Python `Server` API if you need a different host or port.
- `trillim chat` and `trillim serve` take `Trillim/<name>` or `Local/<name>` store IDs, not raw paths.
