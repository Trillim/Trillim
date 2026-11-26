from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path
from threading import Timer
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_FRAME = struct.Struct("<cQ")
_VALUE: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class Response(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    ok: bool
    result: str = ""
    error: str = ""
    kind: Literal["", "invalid", "range", "overflow"] = ""


class NativeCPU:
    def __init__(
        self, executable: Path, errors: Path, environment: dict[str, str]
    ) -> None:
        self.executable = executable
        self.errors = errors
        self.environment = environment
        self.process = self._start()

    def _start(self) -> subprocess.Popen[bytes]:
        with self.errors.open("ab") as stream:
            return subprocess.Popen(
                [str(self.executable)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stream,
                env=self.environment,
            )

    def request(self, mode: bytes, payload: bytes) -> Response:
        if self.process.poll() is not None:
            self.close()
            self.process = self._start()
        process = self.process
        assert process.stdin is not None and process.stdout is not None
        watchdog = Timer(5, process.kill)
        watchdog.daemon = True
        try:
            watchdog.start()
            process.stdin.write(_FRAME.pack(mode, len(payload)) + payload)
            process.stdin.flush()
            response = process.stdout.readline()
            assert response, (
                f"native driver stopped: status={process.poll()}\n"
                f"{self.errors.read_text(errors='replace')[-8000:]}"
            )
            return Response.model_validate_json(response)
        finally:
            watchdog.cancel()
            watchdog.join()

    def call(self, mode: bytes, value: JsonValue) -> Response:
        return self.request(mode, json.dumps(value, allow_nan=False).encode())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        finally:
            if self.process.stdout is not None:
                self.process.stdout.close()


def result(response: Response) -> JsonValue:
    assert response.ok, response.error
    return _VALUE.validate_json(response.result)
