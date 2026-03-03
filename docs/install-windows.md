# Install on Windows

## Requirements

- x86_64 CPU with AVX2
- Python 3.12 or newer

## 1. Check your Python version

In PowerShell:

```powershell
py --version
```

If you see `Python 3.12.x` (or newer), continue to step 3.
If not, install/upgrade Python in step 2.

## 2. Install or upgrade to Python 3.12+

### Option A (recommended): install Python with winget

```powershell
winget install -e --id Python.Python.3.12
py -3.12 --version
```

### Option B: install Python from python.org

Download from [python.org](https://www.python.org/downloads/windows/), run the installer, and check:

- `Add python.exe to PATH`
- `Install launcher for all users (recommended)`

Then verify:

```powershell
py -3.12 --version
```

## 3. Install Trillim with uv (recommended)

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

If you want voice features (`--voice`), install the voice extra:

```powershell
uv add "trillim[voice]"
```

## 4. Verify the install

```powershell
uv run python --version
uv run trillim --help
uv run trillim list
```

## 5. Run Trillim

With `uv`, prefix commands with `uv run`:

```powershell
uv run trillim pull Trillim/BitNet-TRNQ
uv run trillim chat Trillim/BitNet-TRNQ
uv run trillim serve Trillim/BitNet-TRNQ
```

## 6. pip alternative (venv)

If you prefer `pip`:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
pip install trillim
```

Then run:

```powershell
trillim --help
trillim list
```
