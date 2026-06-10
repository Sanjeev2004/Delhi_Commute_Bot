"""
Tests for database CRUD operations (src.db.crud).

All tests use the ``async_db_session`` fixture (in-memory SQLite) so
no external database is required.

Covered operations:
- ``log_query``            — creates a QueryLog row
- Phone number hashing     — SHA-256, never stored as plaintext
- ``update_popular_routes`` — creates or increments PopularRoute
- ``get_stats``            — aggregate analytics
- ``get_popular_routes``   — ordering by count descending
"""

from __future__ import annotations

import hashlib

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.crud import (
    get_popular_routes,
    get_stats,
    log_query,
    update_popular_routes,
)
from src.db.models import PopularRoute, QueryLog


# ---------------------------------------------------------------------------
# log_query tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_query(async_db_session: AsyncSession) -> None:
    """``log_query`` should create a new QueryLog row."""
    entry = await log_query(
        async_db_session,
        raw_query="bus from CP to AIIMS",
        intent="bus_route",
        source_location="Connaught Place",
        destination_location="AIIMS",
        response_text="🚌 DTC Bus: CP → AIIMS",
        response_time_ms=42,
        user_phone="+911234567890",
    )
    await async_db_session.commit()

    assert entry.id is not None, "QueryLog should have an assigned ID"
    assert entry.intent == "bus_route", f"Expected 'bus_route', got '{entry.intent}'"
    assert entry.raw_query == "bus from CP to AIIMS", (
        f"raw_query mismatch: {entry.raw_query}"
    )
    assert entry.response_time_ms == 42, (
        f"response_time_ms should be 42, got {entry.response_time_ms}"
    )


@pytest.mark.asyncio
async def test_log_query_phone_hashed(async_db_session: AsyncSession) -> None:
    """The user phone should be stored as SHA-256 hash, never as plaintext."""
    phone = "+919876543210"
    expected_hash = hashlib.sha256(phone.encode()).hexdigest()

    entry = await log_query(
        async_db_session,
        raw_query="metro to Rajiv Chowk",
        intent="metro_route",
        response_text="🚇 Metro response",
        response_time_ms=15,
        user_phone=phone,
    )
    await async_db_session.commit()

    assert entry.user_phone_hash == expected_hash, (
        f"Phone hash mismatch: expected {expected_hash}, got {entry.user_phone_hash}"
    )
    # Ensure the plaintext phone number is NOT stored anywhere
    assert phone not in (entry.user_phone_hash or ""), (
        "Plaintext phone should never appear in user_phone_hash"
    )
    assert phone not in (entry.raw_query or ""), (
        "Plaintext phone should not be in raw_query"
    )


# ---------------------------------------------------------------------------
# update_popular_routes tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_popular_routes_new(async_db_session: AsyncSession) -> None:
    """First call should create a new PopularRoute with count=1."""
    route = await update_popular_routes(
        async_db_session, "Dwarka", "Rajiv Chowk"
    )
    await async_db_session.commit()

    assert route.query_count == 1, (
        f"New route should have count=1, got {route.query_count}"
    )
    assert route.source.strip().lower() == "dwarka", (
        f"Source mismatch: expected 'dwarka', got '{route.source}'"
    )
    assert route.destination.strip().lower() == "rajiv chowk", (
        f"Destination mismatch: expected 'rajiv chowk', got '{route.destination}'"
    )


@pytest.mark.asyncio
async def test_update_popular_routes_increment(async_db_session: AsyncSession) -> None:
    """Calling update_popular_routes again for the same pair should increment."""
    # First call → creates
    await update_popular_routes(async_db_session, "Nehru Place", "AIIMS")
    await async_db_session.commit()

    # Second call → increments
    route = await update_popular_routes(async_db_session, "Nehru Place", "AIIMS")
    await async_db_session.commit()

    assert route.query_count == 2, (
        f"Expected count=2 after two calls, got {route.query_count}"
    )


# ---------------------------------------------------------------------------
# get_stats tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_stats(async_db_session: AsyncSession) -> None:
    """``get_stats`` should return correct aggregate data."""
    # Seed some data
    await log_query(
        async_db_session,
        raw_query="bus from A to B",
        intent="bus_route",
        response_text="response1",
        response_time_ms=10,
        user_phone="+911111111111",
    )
    await log_query(
        async_db_session,
        raw_query="metro from C to D",
        intent="metro_route",
        response_text="response2",
        response_time_ms=20,
        user_phone="+912222222222",
    )
    await async_db_session.commit()

    stats = await get_stats(async_db_session)

    assert stats["total_queries"] >= 2, (
        f"Expected at least 2 total_queries, got {stats['total_queries']}"
    )
    assert "queries_today" in stats, "Stats should include 'queries_today'"
    assert isinstance(stats["queries_today"], int), (
        f"queries_today should be int, got {type(stats['queries_today'])}"
    )
    assert "top_intents" in stats, "Stats should include 'top_intents'"
    assert isinstance(stats["top_intents"], dict), (
        f"top_intents should be dict, got {type(stats['top_intents'])}"
    )


# ---------------------------------------------------------------------------
# get_popular_routes ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_popular_routes_ordering(async_db_session: AsyncSession) -> None:
    """Popular routes should be returned in descending order by query_count."""
    # Create routes with different counts
    await update_popular_routes(async_db_session, "Low Route Src", "Low Route Dst")
    await async_db_session.commit()

    # This route gets 3 total calls → highest count
    await update_popular_routes(async_db_session, "High Route Src", "High Route Dst")
    await async_db_session.commit()
    await update_popular_routes(async_db_session, "High Route Src", "High Route Dst")
    await async_db_session.commit()
    await update_popular_routes(async_db_session, "High Route Src", "High Route Dst")
    await async_db_session.commit()

    # This route gets 2 calls → medium count
    await update_popular_routes(async_db_session, "Mid Route Src", "Mid Route Dst")
    await async_db_session.commit()
    await update_popular_routes(async_db_session, "Mid Route Src", "Mid Route Dst")
    await async_db_session.commit()

    routes = await get_popular_routes(async_db_session, limit=10)

    assert len(routes) >= 3, f"Expected at least 3 routes, got {len(routes)}"

    # Verify descending order
    counts = [r.query_count for r in routes]
    assert counts == sorted(counts, reverse=True), (
        f"Routes should be ordered by count descending, got {counts}"
    )

    # The first route should be the one with count=3
    assert routes[0].source.strip().lower() == "high route src", (
        f"Expected 'High Route Src' first, got '{routes[0].source}'"
    )
