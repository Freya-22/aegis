"""Shape of an OpenAI-style chat request.

Anything that doesn't match is rejected with HTTP 422 before it reaches the model.
The size limits are a cheap first defence against oversized prompts that burn
GPU time (OWASP LLM10: Unbounded Consumption).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_MESSAGES = 50
MAX_CONTENT_CHARS = 20_000


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    # Plain text only for now: the Phase 3 guards scan text.
    content: str = Field(max_length=MAX_CONTENT_CHARS)


class ChatCompletionRequest(BaseModel):
    # Unknown OpenAI options (top_p, seed, ...) are kept and passed through to the backend.
    model_config = ConfigDict(
        extra="allow",
        # Pre-filled body on the /docs page; without it the page suggests max_tokens=0, which fails validation.
        json_schema_extra={"examples": [{
            "model": "llama3.2:3b",
            "messages": [{"role": "user", "content": "In one sentence, what is a reverse proxy?"}],
        }]},
    )

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0, le=4096)
    stream: bool = False
