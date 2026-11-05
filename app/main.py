"""Aegis gateway entry point.

Run:  uvicorn app.main:app --port 8080
Any OpenAI-compatible client can use it by setting base_url=http://localhost:8080/v1
"""

import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request

from app.backend import BackendError, OllamaBackend
from app.config import settings
from app.schemas import ChatCompletionRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=settings.backend_timeout)
    app.state.backend = OllamaBackend(client)
    yield
    await client.aclose()


app = FastAPI(title="Aegis", version="0.2.0", lifespan=lifespan)


@app.middleware("http")
async def tag_request(request: Request, call_next):
    """Give every request an ID and measure how long Aegis takes.

    The ID ties together all events for one request once telemetry lands (Phase 4).
    """
    request.state.request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Aegis-Latency-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    if body.stream:
        # The outbound guard (Phase 3) must see the whole answer before releasing it.
        raise HTTPException(400, "Streaming is not supported by this gateway")

    model = body.model or settings.default_model
    if model not in settings.allowed_models:
        raise HTTPException(400, f"Model '{model}' is not allowed")

    payload = body.model_dump(exclude_none=True)
    payload["model"] = model

    # Phase 3: inbound guards (prompt injection, PII redaction) go here.
    try:
        result = await request.app.state.backend.chat(payload)
    except BackendError as e:
        raise HTTPException(e.status_code, e.detail)
    # Phase 3: outbound guard (secret-leak scan) goes here.

    return result
