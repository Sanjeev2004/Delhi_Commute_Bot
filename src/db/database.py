"""
Async database engine and session management.

Provides:
- Async SQLAlchemy engine (PostgreSQL primary, SQLite fallback for dev)
- AsyncSession factory via ``async_session_factory``
- ``get_db()`` FastAPI dependency yielding sessions
- ``init_db()`` to create all tables on startup
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from loguru import logger

from src.config import settings
from src.db.models import Base


def _resolve_database_url() -> str:
    """Return the best available database URL.

    Tries the configured PostgreSQL URL first.  When the ``asyncpg`` driver
    cannot be imported (i.e. PostgreSQL is unavailable), falls back to a
    local SQLite file so development can proceed without Docker.
    """
    pg_url = settings.database_url

    if "sqlite" in pg_url:
        # Already configured as SQLite – honour it.
        return pg_url

    try:
        import asyncpg  # noqa: F401 – availability check

        logger.info("Using PostgreSQL database: {}", pg_url)
        return pg_url
    except ImportError:
        pass

    # Fallback to SQLite (aiosqlite driver).
    from pathlib import Path

    db_path = Path(settings.faiss_index_path).parent / "delhicommutebot.db"
    sqlite_url = f"sqlite+aiosqlite:///{db_path}"
    logger.warning(
        "asyncpg not available – falling back to SQLite at {}",
        db_path,
    )
    return sqlite_url


_database_url: str = _resolve_database_url()

engine = create_async_engine(
    _database_url,
    echo=settings.is_development,
    pool_pre_ping=True,
    # SQLite does not support pool_size / max_overflow
    **({} if "sqlite" in _database_url else {"pool_size": 5, "max_overflow": 10}),
)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:  # type: ignore[misc]
    """FastAPI dependency that yields an ``AsyncSession``.

    The session is committed on success and rolled back on any exception.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables defined in :pymod:`src.db.models`.

    Safe to call repeatedly – SQLAlchemy's ``create_all`` is idempotent.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialised.")
