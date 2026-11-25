"""Application entrypoint.

Waits for database, creates tables, then launches Gradio.
"""

from __future__ import annotations

import logging
import sys

from app.config import get_settings, setup_logging
from app.database import init_db, wait_for_db
from app.ui.gradio_app import build_ui

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    settings = get_settings()
    logger.info("starting Team Reporting Agent")

    try:
        settings.validate_for_db()
        wait_for_db()
        init_db()
    except Exception as exc:
        logger.error("database startup failed: %s", exc)
        sys.exit(1)

    settings = get_settings()
    logger.info(
        "LLM endpoint: %s  model: %s  (local=%s)",
        settings.llm_base_url,
        settings.llm_model,
        settings.is_local_llm,
    )
    if not settings.is_local_llm and not settings.openrouter_api_key:
        logger.warning(
            "No API key set for cloud provider — report generation and agent chat will fail"
        )

    demo = build_ui()
    demo.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        share=False,
    )


if __name__ == "__main__":
    main()
