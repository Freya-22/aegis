"""Gateway settings, loaded once from environment variables / .env."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    default_model: str
    # Only models on this list can be called through the gateway.
    allowed_models: frozenset[str]
    # Seconds to wait for the model before giving up (a 3B model on a laptop can be slow).
    backend_timeout: float


def load_settings() -> Settings:
    default_model = os.getenv("DEFAULT_MODEL", "llama3.2:3b")
    allowed = os.getenv("ALLOWED_MODELS", default_model)
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        default_model=default_model,
        allowed_models=frozenset(m.strip() for m in allowed.split(",") if m.strip()),
        backend_timeout=float(os.getenv("BACKEND_TIMEOUT", "120")),
    )


settings = load_settings()
