"""Application configuration loaded from environment variables.

Supports both OpenRouter (cloud) and local llama.cpp / any OpenAI-compatible server.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Settings:
    """Central settings object. Values come from env vars / .env."""

    def __init__(self) -> None:
        # ---- Database ----
        # Prefer explicit DATABASE_URL. If unset, fall back to SQLite for easy local use.
        default_sqlite = "sqlite+pysqlite:///./data/daily_reports.db"
        self.database_url: str = os.getenv("DATABASE_URL", default_sqlite)

        # ---- LLM (OpenAI-compatible) ----
        # Works with:
        #   - llama.cpp server  (http://localhost:8080/v1)
        #   - Ollama            (http://localhost:11434/v1)
        #   - LM Studio         (http://localhost:1234/v1)
        #   - OpenRouter        (https://openrouter.ai/api/v1)
        #   - any other OpenAI-compatible endpoint
        self.llm_base_url: str = os.getenv(
            "LLM_BASE_URL",
            os.getenv("OPENROUTER_BASE_URL", "http://localhost:8080/v1"),
        )
        # API key is optional for many local servers; OpenRouter needs a real key.
        self.llm_api_key: str = (
            os.getenv("LLM_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or "local-no-key"
        )
        self.llm_model: str = (
            os.getenv("LLM_MODEL")
            or os.getenv("OPENROUTER_MODEL")
            or "local-model"
        )

        # Backward-compat aliases (used by older code paths)
        self.openrouter_api_key: str | None = (
            None if self.llm_api_key in ("", "local-no-key") else self.llm_api_key
        )
        self.openrouter_model: str = self.llm_model
        self.openrouter_base_url: str = self.llm_base_url

        # ---- Gradio ----
        self.gradio_server_name: str = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
        self.gradio_server_port: int = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

        # ---- Behaviour ----
        self.force_tools: bool = os.getenv("FORCE_TOOLS", "1").lower() in ("1", "true", "yes")

    @property
    def is_local_llm(self) -> bool:
        url = (self.llm_base_url or "").lower()
        return any(
            host in url
            for host in ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal")
        )

    def validate_for_llm(self) -> None:
        """Soft validation – local servers usually need no key."""
        if not self.llm_base_url:
            raise RuntimeError(
                "LLM_BASE_URL is not set. "
                "Example for llama.cpp: LLM_BASE_URL=http://localhost:8080/v1"
            )
        if not self.is_local_llm and (
            not self.llm_api_key or self.llm_api_key == "local-no-key"
        ):
            raise RuntimeError(
                "A real API key is required for cloud providers. "
                "Set LLM_API_KEY or OPENROUTER_API_KEY."
            )

    def validate_for_db(self) -> None:
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is not set.")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def setup_logging(level: str | None = None) -> None:
    """Configure root logger once at startup."""
    lvl = (level or get_settings().log_level).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_llm_client():
    """Return an OpenAI client pointed at the configured endpoint (local or cloud)."""
    from openai import OpenAI

    settings = get_settings()
    settings.validate_for_llm()
    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
