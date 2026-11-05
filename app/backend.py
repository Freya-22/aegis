"""Client for the model server (Ollama's OpenAI-compatible API)."""

import httpx


class BackendError(Exception):
    """The model server failed or could not be reached."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class OllamaBackend:
    def __init__(self, client: httpx.AsyncClient):
        # One shared client reuses connections instead of opening a new one per request.
        self.client = client

    async def chat(self, payload: dict) -> dict:
        try:
            resp = await self.client.post("/v1/chat/completions", json=payload)
        except httpx.TimeoutException:
            raise BackendError(504, "Model backend timed out")
        except httpx.HTTPError:
            raise BackendError(502, "Model backend unreachable")

        if resp.status_code != 200:
            # Don't echo the backend's raw error to the caller: it can expose internals.
            raise BackendError(502, f"Model backend returned HTTP {resp.status_code}")
        return resp.json()
