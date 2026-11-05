"""Proxy tests. A fake Ollama stands in for the real one, so no model or GPU is needed."""

import httpx
import pytest
from fastapi.testclient import TestClient

from app.backend import OllamaBackend
from app.main import app

FAKE_REPLY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "llama3.2:3b",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Aegis online"}, "finish_reason": "stop"}],
}


def make_client(handler):
    """TestClient whose backend is replaced by `handler` (a fake Ollama)."""
    fake = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    client = TestClient(app)
    client.__enter__()  # run startup, then swap in the fake backend
    app.state.backend = OllamaBackend(fake)
    return client


@pytest.fixture
def seen():
    return []


@pytest.fixture
def client(seen):
    def handler(request: httpx.Request):
        seen.append(request)
        return httpx.Response(200, json=FAKE_REPLY)

    c = make_client(handler)
    yield c
    c.__exit__(None, None, None)


def chat(client, **overrides):
    body = {"messages": [{"role": "user", "content": "hi"}], **overrides}
    return client.post("/v1/chat/completions", json=body)


def test_forwards_and_returns_answer(client, seen):
    r = chat(client)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Aegis online"
    assert seen[0].url.path == "/v1/chat/completions"


def test_fills_in_default_model(client, seen):
    chat(client)
    import json
    assert json.loads(seen[0].content)["model"] == "llama3.2:3b"


def test_adds_request_id_and_latency_headers(client):
    r = chat(client)
    assert len(r.headers["X-Request-ID"]) == 12
    assert float(r.headers["X-Aegis-Latency-Ms"]) >= 0


def test_rejects_unlisted_model(client, seen):
    r = chat(client, model="some-other-model")
    assert r.status_code == 400
    assert seen == []  # never reached the backend


def test_rejects_streaming(client):
    assert chat(client, stream=True).status_code == 400


@pytest.mark.parametrize("bad", [
    {"messages": []},                                             # no messages
    {"messages": [{"role": "hacker", "content": "hi"}]},          # unknown role
    {"messages": [{"role": "user", "content": "x" * 20_001}]},    # oversized prompt
    {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 999_999},
])
def test_rejects_malformed_requests(client, seen, bad):
    assert client.post("/v1/chat/completions", json=bad).status_code == 422
    assert seen == []


def test_backend_down_returns_502():
    def handler(request):
        raise httpx.ConnectError("refused")

    c = make_client(handler)
    r = chat(c)
    c.__exit__(None, None, None)
    assert r.status_code == 502


def test_backend_error_is_not_leaked():
    def handler(request):
        return httpx.Response(500, text="internal stack trace: /opt/secret/path")

    c = make_client(handler)
    r = chat(c)
    c.__exit__(None, None, None)
    assert r.status_code == 502
    assert "secret" not in r.text
