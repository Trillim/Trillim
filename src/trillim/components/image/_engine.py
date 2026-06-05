"""Managed image-generation subprocess protocol."""

from __future__ import annotations

import asyncio
import binascii
import os
import struct
import tempfile
import zlib
from collections.abc import Callable
from pathlib import Path

from trillim.components.llm._config import ModelRuntimeConfig
from trillim.components.llm._engine import (
    EngineCrashedError,
    EngineError,
    EngineProgressTimeoutError,
    _first_protocol_line,
    _read_stderr,
)

_SEQ_LEN_BUCKETS = (32, 64, 128, 256, 512)


class ImageEngine:
    """Drive the Bonsai Image worker for one active model."""

    def __init__(
        self,
        model: ModelRuntimeConfig,
        tokenizer,
        *,
        num_threads: int = 0,
        progress_timeout: float = 600.0,
        _binary_path: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.num_threads = num_threads
        self.progress_timeout = progress_timeout
        self.binary_path = (
            _bundled_image_binary_path() if _binary_path is None else _binary_path
        )
        self.process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        """Start the image worker and send its init block."""
        if self.process is not None and self.process.returncode is None:
            return
        self.process = await asyncio.create_subprocess_exec(
            self.binary_path,
            str(self.model.path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await self._write_block(
                _build_image_init_block(self.model, self.num_threads)
            )
        except Exception:
            await self._kill_process()
            raise

    async def stop(self) -> None:
        """Stop the image worker."""
        process = self.process
        self.process = None
        if process is None or process.returncode is not None:
            return
        try:
            assert process.stdin is not None
            process.stdin.write(b"0\n")
            await asyncio.wait_for(process.stdin.drain(), timeout=self.progress_timeout)
            await asyncio.wait_for(process.wait(), timeout=self.progress_timeout)
        except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError, OSError):
            try:
                process.kill()
            except ProcessLookupError:
                return
            await process.wait()

    async def generate(
        self,
        prompt: str,
        output_path: str | Path,
        *,
        steps: int,
        seed: int | None,
        width: int,
        height: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Generate one PNG image."""
        self._require_running()
        token_ids, attention_mask = _encode_prompt_tokens(self.tokenizer, prompt)
        output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        raw_path = _temporary_raw_path(output)
        try:
            await self._write_block(
                _build_image_request_block(
                    output_path=raw_path,
                    token_ids=token_ids,
                    attention_mask=attention_mask,
                    steps=steps,
                    seed=0 if seed is None else seed,
                    width=width,
                    height=height,
                )
            )
            while True:
                status = (
                    (await self._readline("image_status"))
                    .decode("utf-8", errors="replace")
                    .strip()
                )
                progress = _parse_progress_status(status)
                if progress is None:
                    break
                if progress_callback is not None:
                    progress_callback(*progress)

            detail = (
                (await self._readline("image_detail"))
                .decode("utf-8", errors="replace")
                .strip()
            )
            if status != "ok":
                await self._kill_process()
                raise EngineError(detail or "Image generation failed")
            rgb = raw_path.read_bytes()
            expected = width * height * 3
            if len(rgb) != expected:
                await self._kill_process()
                raise EngineError(
                    f"Image worker wrote {len(rgb)} RGB bytes; expected {expected}"
                )
            _write_png(output, width, height, rgb)
            return output
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            await self._kill_process()
            raise EngineCrashedError("Image engine crashed") from exc
        finally:
            try:
                raw_path.unlink()
            except FileNotFoundError:
                pass

    def _require_running(self) -> asyncio.subprocess.Process:
        process = self.process
        if process is None or process.returncode is not None:
            raise EngineCrashedError("Image process is not running")
        return process

    async def _write_block(self, block: str) -> None:
        process = self._require_running()
        stdin = process.stdin
        if stdin is None:
            raise EngineCrashedError("Image process stdin is unavailable")
        stdin.write(block.encode("utf-8"))
        try:
            await asyncio.wait_for(stdin.drain(), timeout=self.progress_timeout)
        except asyncio.TimeoutError as exc:
            raise EngineProgressTimeoutError(
                f"Image engine made no write progress for {self.progress_timeout} seconds"
            ) from exc

    async def _readline(self, field_name: str) -> bytes:
        process = self._require_running()
        stdout = process.stdout
        if stdout is None:
            raise EngineCrashedError("Image process stdout is unavailable")
        try:
            line = await asyncio.wait_for(
                stdout.readline(),
                timeout=self.progress_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise EngineProgressTimeoutError(
                f"Image engine made no {field_name} progress for {self.progress_timeout} seconds"
            ) from exc
        if not line:
            stderr = await _read_stderr(process)
            if stderr:
                raise EngineCrashedError(f"Image engine crashed: {stderr}")
            raise EngineCrashedError("Image engine crashed")
        return line

    async def _kill_process(self) -> None:
        process = self.process
        if process is None:
            return
        if process.returncode is not None:
            self.process = None
            return
        try:
            process.kill()
        except OSError:
            pass
        await process.wait()
        self.process = None


def _bundled_image_binary_path() -> str:
    suffix = ".exe" if os.name == "nt" else ""
    bin_dir = Path(__file__).resolve().parents[2] / "_bin"
    bundled = bin_dir / f"trillim-image-inference{suffix}"
    if bundled.is_file():
        return str(bundled)
    if suffix:
        fallback = bin_dir / "trillim-image-inference"
        if fallback.is_file():
            return str(fallback)
    raise FileNotFoundError(f"Missing bundled image inference binary: {bundled}")


def _build_image_init_block(model: ModelRuntimeConfig, num_threads: int) -> str:
    pairs = [f"arch_type={int(model.arch_type)}"]
    if num_threads:
        pairs.append(f"num_threads={num_threads}")
    return f"{len(pairs)}\n" + "\n".join(pairs) + "\n"


def _build_image_request_block(
    *,
    output_path: Path,
    token_ids: list[int],
    attention_mask: list[int],
    steps: int,
    seed: int,
    width: int,
    height: int,
) -> str:
    pairs = [
        f"output_path={_first_protocol_line(str(output_path))}",
        f"tokens={','.join(str(token_id) for token_id in token_ids)}",
        f"attention_mask={','.join(str(value) for value in attention_mask)}",
        f"steps={steps}",
        f"seed={seed}",
        f"width={width}",
        f"height={height}",
    ]
    return f"{len(pairs)}\n" + "\n".join(pairs) + "\n"


def _parse_progress_status(status: str) -> tuple[int, int] | None:
    parts = status.split()
    if len(parts) != 3 or parts[0] != "progress":
        return None
    try:
        done = int(parts[1])
        total = int(parts[2])
    except ValueError:
        return None
    if done < 0 or total < 1:
        return None
    return min(done, total), total


def _encode_prompt_tokens(tokenizer, prompt: str) -> tuple[list[int], list[int]]:
    formatted_prompt = prompt
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        formatted_prompt = apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    unpadded = tokenizer(
        [formatted_prompt],
        padding=False,
        truncation=False,
        add_special_tokens=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors=None,
    )
    true_len = len(unpadded["input_ids"][0])
    max_length = _pick_sequence_bucket(true_len)
    tokens = tokenizer(
        [formatted_prompt],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
        return_tensors=None,
    )
    return _first_tokenizer_row(tokens["input_ids"]), _first_tokenizer_row(
        tokens["attention_mask"]
    )


def _pick_sequence_bucket(true_len: int) -> int:
    for bucket in _SEQ_LEN_BUCKETS:
        if bucket >= true_len:
            return bucket
    return _SEQ_LEN_BUCKETS[-1]


def _first_tokenizer_row(value) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and hasattr(value[0], "tolist"):
        value = value[0].tolist()
    elif value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def _temporary_raw_path(output_path: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".rgb",
        dir=output_path.parent,
        delete=False,
    ) as handle:
        return Path(handle.name)


def _write_png(path: Path, width: int, height: int, rgb: bytes) -> None:
    if width < 1 or height < 1:
        raise ValueError("PNG dimensions must be positive")
    row_bytes = width * 3
    if len(rgb) != row_bytes * height:
        raise ValueError("RGB payload size does not match PNG dimensions")
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        start = y * row_bytes
        rows.extend(rgb[start : start + row_bytes])

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
        )

    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows), level=6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(payload)
