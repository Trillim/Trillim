"""HTTP router for the LLM component."""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.types import Receive, Scope, Send

from trillim.components.llm._config import ModelInfo
from trillim.components.llm._engine import (
    EngineCrashedError,
    EngineError,
    EngineProgressTimeoutError,
)
from trillim.components.llm._events import ChatDoneEvent, ChatTokenEvent
from trillim.components.llm._limits import REQUEST_BODY_LIMIT_BYTES
from trillim.components.llm._validation import validate_chat_request, validate_swap_request
from trillim.errors import (
    AdmissionRejectedError,
    ContextOverflowError,
    InvalidRequestError,
    ModelValidationError,
    ProgressTimeoutError,
    SessionClosedError,
    SessionExhaustedError,
    SessionStaleError,
)


class _ChatStreamResponse(Response):
    media_type = "text/event-stream"

    def __init__(self, llm, request_model) -> None:
        self._llm = llm
        self._request_model = request_model
        self.status_code = 200
        self.background = None
        self.body = None
        self.init_headers()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        session = None
        admission_lease = None
        try:
            try:
                session = self._llm.open_session(_request_messages(self._request_model))
                sampling = session._prepare_stream_chat(**_sampling_kwargs(self._request_model))
                admission_lease = await self._llm._admission.acquire()
            except Exception as exc:
                try:
                    await _http_exception_response(_as_http_error(exc))(scope, receive, send)
                except OSError:
                    pass
                return

            response_id = _response_id()
            created = int(time.time())
            try:
                await send(
                    {
                        "type": "http.response.start",
                        "status": self.status_code,
                        "headers": self.raw_headers,
                    }
                )
                async with session:
                    async for chunk in _stream_chat_response(
                        self._llm,
                        session,
                        sampling=sampling,
                        response_id=response_id,
                        created=created,
                    ):
                        await send(
                            {
                                "type": "http.response.body",
                                "body": chunk.encode("utf-8"),
                                "more_body": True,
                            }
                        )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"",
                        "more_body": False,
                    }
                )
            except EngineProgressTimeoutError as exc:
                await admission_lease.release()
                admission_lease = None
                await self._llm._recover_from_engine_failure()
                raise ProgressTimeoutError(str(exc)) from exc
            except (EngineCrashedError, EngineError) as exc:
                await admission_lease.release()
                admission_lease = None
                await self._llm._recover_from_engine_failure()
                raise RuntimeError(str(exc)) from exc
            except OSError:
                return
        finally:
            if admission_lease is not None:
                await admission_lease.release()
            if session is not None:
                await session.close()


def build_router(llm, *, allow_hot_swap: bool) -> APIRouter:
    """Build the HTTP router for an LLM component instance."""
    router = APIRouter()

    @router.get("/v1/models")
    async def list_models():
        info = llm.model_info()
        return {
            "object": "list",
            "state": info.state,
            "data": [] if info.name is None else [_model_payload(info)],
        }

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        info = llm.model_info()
        payload = await _read_json_body(request, REQUEST_BODY_LIMIT_BYTES)
        try:
            chat_request = validate_chat_request(
                payload,
                active_model_name=info.name,
            )
            if chat_request.stream:
                return _ChatStreamResponse(llm, chat_request)
            text, usage = await llm._collect_chat(
                _request_messages(chat_request),
                **_sampling_kwargs(chat_request),
            )
        except Exception as exc:
            raise _as_http_error(exc) from exc
        model_info = llm.model_info()
        response_id = _response_id()
        created = int(time.time())
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model_info.name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cached_tokens": usage.cached_tokens,
            },
        }

    if allow_hot_swap:

        @router.post("/v1/models/swap")
        async def swap_model_route(request: Request):
            payload = await _read_json_body(request, REQUEST_BODY_LIMIT_BYTES)
            try:
                swap_request = validate_swap_request(payload)
                info = await llm.swap_model(
                    swap_request.model_dir,
                    num_threads=swap_request.num_threads,
                    lora_dir=swap_request.lora_dir,
                    lora_quant=swap_request.lora_quant,
                    unembed_quant=swap_request.unembed_quant,
                    harness_name=swap_request.harness_name,
                    search_provider=swap_request.search_provider,
                    search_token_budget=swap_request.search_token_budget,
                )
            except Exception as exc:
                raise _as_http_error(exc) from exc
            return {
                "object": "model.swap",
                "model": info.name,
                "state": info.state,
                "path": info.path,
                "max_context_tokens": info.max_context_tokens,
                "trust_remote_code": info.trust_remote_code,
                "adapter_path": info.adapter_path,
                "init_config": _init_config_payload(info),
            }

    return router


async def _stream_chat_response(llm, session, *, sampling, response_id: str, created: int):
    yield _sse(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": llm.model_info().name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
    )
    async for event in session._stream_chat_prepared(sampling):
        if isinstance(event, ChatTokenEvent):
            if not event.text:
                continue
            yield _sse(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": llm.model_info().name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": event.text},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        elif isinstance(event, ChatDoneEvent):
            yield _sse(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": llm.model_info().name,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                }
            )
            break
    yield "data: [DONE]\n\n"


def _model_payload(info: ModelInfo) -> dict[str, object]:
    return {
        "id": info.name,
        "object": "model",
        "path": info.path,
        "max_context_tokens": info.max_context_tokens,
        "trust_remote_code": info.trust_remote_code,
        "adapter_path": info.adapter_path,
        "init_config": _init_config_payload(info),
    }


def _init_config_payload(info: ModelInfo) -> dict[str, object] | None:
    if info.init_config is None:
        return None
    return {
        "num_threads": info.init_config.num_threads,
        "lora_quant": info.init_config.lora_quant,
        "unembed_quant": info.init_config.unembed_quant,
    }


async def _read_json_body(request: Request, limit: int) -> object:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > limit:
                raise HTTPException(status_code=413, detail="request body too large")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid content-length header") from exc
    total = 0
    chunks: list[bytes] = []
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="request body too large")
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc


def _as_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, (InvalidRequestError, ModelValidationError, ContextOverflowError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, (SessionClosedError, SessionExhaustedError, SessionStaleError)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, AdmissionRejectedError):
        return HTTPException(status_code=429, detail=str(exc))
    if isinstance(exc, ProgressTimeoutError):
        return HTTPException(status_code=504, detail=str(exc))
    return HTTPException(status_code=503, detail=str(exc))


def _http_exception_response(exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,
    )


def _request_messages(request_model) -> list[dict[str, str]]:
    return [
        {"role": message.role, "content": message.content}
        for message in request_model.messages
    ]


def _sampling_kwargs(request_model) -> dict[str, float | int | None]:
    return {
        "temperature": request_model.temperature,
        "top_k": request_model.top_k,
        "top_p": request_model.top_p,
        "repetition_penalty": request_model.repetition_penalty,
        "rep_penalty_lookback": request_model.rep_penalty_lookback,
        "max_tokens": request_model.max_tokens,
    }


def _response_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"
