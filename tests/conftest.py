"""
Shared test fixtures for the DelhiCommuteBot test suite.

Provides:
- ``async_db_session`` — In-memory SQLite async session with all tables created
- ``app_client``        — httpx.AsyncClient wired to the FastAPI test app
- ``sample_query_request`` — Reusable sample payload for /query
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.db.models import Base
from src.db.database import get_db
from src.main import app


# ---------------------------------------------------------------------------
# Database fixture — in-memory SQLite
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create an in-memory SQLite async session for testing.

    - Uses the ``aiosqlite`` driver so no real DB is required.
    - Creates **all** ORM tables before yielding the session.
    - Cleans up (drops tables, disposes engine) after the test.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ---------------------------------------------------------------------------
# FastAPI test client — overrides the DB dependency
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def app_client(
    async_db_session: AsyncSession,
) -> AsyncGenerator[AsyncClient, None]:
    """httpx.AsyncClient wired to the FastAPI test application.

    Overrides ``get_db`` so every request uses the in-memory SQLite
    session instead of the production database.
    """

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield async_db_session

    app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    # Remove override after test
    app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_query_request() -> dict:
    """Reusable payload for the ``POST /query`` endpoint."""
    return {
        "raw_text": "bus from CP to AIIMS",
        "user_phone": "+911234567890",
    }
