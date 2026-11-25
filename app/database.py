"""Database engine, session factory, and connection retry logic.

Supports PostgreSQL (production / Docker) and SQLite (easy local runs).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def get_engine(url: str | None = None, *, echo: bool = False):
    """Create (or return cached) SQLAlchemy engine."""
    global _engine, _SessionLocal
    if _engine is not None and url is None:
        return _engine

    settings = get_settings()
    db_url = url or settings.database_url

    # Ensure parent directory exists for SQLite file databases
    if _is_sqlite(db_url):
        path_part = db_url.split("///")[-1]
        if path_part and path_part != ":memory:":
            Path(path_part).parent.mkdir(parents=True, exist_ok=True)

    connect_args = {}
    if _is_sqlite(db_url):
        connect_args["check_same_thread"] = False

    engine = create_engine(
        db_url,
        echo=echo,
        pool_pre_ping=True,
        connect_args=connect_args,
        **({} if _is_sqlite(db_url) else {"pool_size": 5, "max_overflow": 10}),
    )
    if url is None:
        _engine = engine
        _SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return engine


def get_session_factory(url: str | None = None) -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is not None and url is None:
        return _SessionLocal
    engine = get_engine(url)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    if url is None:
        _SessionLocal = factory
    return factory


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager that yields a session and commits / rolls back."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def wait_for_db(max_retries: int = 30, delay: float = 2.0) -> None:
    """Block until the database accepts connections.

    For SQLite this is essentially a no-op (file is created on first connect).
    For PostgreSQL it retries until the server is ready (Docker entrypoint).
    """
    settings = get_settings()
    settings.validate_for_db()

    if _is_sqlite(settings.database_url):
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("SQLite database ready")
        return

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            engine = get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("database connection established (attempt %d)", attempt)
            return
        except Exception as exc:
            last_err = exc
            logger.warning(
                "database not ready (attempt %d/%d): %s",
                attempt,
                max_retries,
                exc,
            )
            time.sleep(delay)
    raise RuntimeError(
        f"Could not connect to PostgreSQL after {max_retries} attempts: {last_err}"
    )


def _ensure_report_path_column(engine) -> None:
    """Add reports.path if missing (existing DBs created before path support)."""
    from sqlalchemy import inspect, text as sa_text

    try:
        insp = inspect(engine)
        if "reports" not in insp.get_table_names():
            return
        cols = {c["name"] for c in insp.get_columns("reports")}
        if "path" not in cols:
            with engine.begin() as conn:
                conn.execute(sa_text("ALTER TABLE reports ADD COLUMN path VARCHAR(128)"))
            logger.info("added reports.path column")
    except Exception as exc:
        logger.warning("path column migration skipped: %s", exc)


def init_db() -> None:
    """Create tables if they do not exist; seed default progress paths."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_report_path_column(engine)
    try:
        from app.reports.progress import ensure_default_paths

        with get_db() as session:
            ensure_default_paths(session)
    except Exception as exc:
        logger.warning("default path seed skipped: %s", exc)
    logger.info("database tables ensured")


def reset_engine() -> None:
    """Reset cached engine (useful in tests)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
