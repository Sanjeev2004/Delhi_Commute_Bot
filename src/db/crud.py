"""
Async CRUD operations for DelhiCommuteBot.

All functions accept an ``AsyncSession`` as their first argument so
they compose naturally with the ``get_db()`` FastAPI dependency.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.db.models import QueryLog, PopularRoute


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_phone(phone: str | None) -> str:
    """Return a SHA-256 hex digest of *phone*, or ``'anonymous'``."""
    if not phone:
        return "anonymous"
    return hashlib.sha256(phone.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Query logging
# ---------------------------------------------------------------------------

async def log_query(
    db: AsyncSession,
    *,
    raw_query: str,
    intent: str,
    source_location: str | None = None,
    destination_location: str | None = None,
    response_text: str,
    response_time_ms: int,
    user_phone: str | None = None,
) -> QueryLog:
    """Insert a new :class:`QueryLog` row and return it."""
    entry = QueryLog(
        user_phone_hash=_hash_phone(user_phone),
        raw_query=raw_query,
        intent=intent,
        source_location=source_location or "",
        destination_location=destination_location or "",
        response_text=response_text,
        response_time_ms=response_time_ms,
    )
    db.add(entry)
    await db.flush()
    logger.debug("Logged query id={}", entry.id)
    return entry


# ---------------------------------------------------------------------------
# Popular routes
# ---------------------------------------------------------------------------

async def update_popular_routes(
    db: AsyncSession,
    source: str,
    destination: str,
) -> PopularRoute:
    """Upsert the popular-route counter for *source* → *destination*.

    If the pair already exists the counter is incremented; otherwise a
    new row is created with ``query_count=1``.
    """
    source_lower = source.strip().lower()
    dest_lower = destination.strip().lower()

    stmt = select(PopularRoute).where(
        and_(
            func.lower(PopularRoute.source) == source_lower,
            func.lower(PopularRoute.destination) == dest_lower,
        )
    )
    result = await db.execute(stmt)
    route = result.scalar_one_or_none()

    if route is not None:
        route.query_count += 1
        route.last_queried = datetime.now(timezone.utc)
        logger.debug(
            "Incremented popular route {} → {} (count={})",
            source, destination, route.query_count,
        )
    else:
        route = PopularRoute(
            source=source.strip(),
            destination=destination.strip(),
            query_count=1,
        )
        db.add(route)
        logger.debug("Created popular route {} → {}", source, destination)

    await db.flush()
    return route


async def get_popular_routes(
    db: AsyncSession,
    limit: int = 10,
) -> list[PopularRoute]:
    """Return the top *limit* most-queried routes ordered by count desc."""
    stmt = (
        select(PopularRoute)
        .order_by(PopularRoute.query_count.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Analytics / stats
# ---------------------------------------------------------------------------

async def get_stats(db: AsyncSession) -> dict[str, Any]:
    """Return aggregate statistics for the dashboard.

    Keys returned:
    - ``total_queries``: int
    - ``queries_today``: int
    - ``top_intents``: dict mapping intent → count
    - ``top_routes``: list of dicts ``{source, destination, count}``
    """
    # Total queries
    total_result = await db.execute(select(func.count(QueryLog.id)))
    total_queries: int = total_result.scalar() or 0

    # Queries today (UTC)
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    today_result = await db.execute(
        select(func.count(QueryLog.id)).where(
            QueryLog.created_at >= today_start,
        )
    )
    queries_today: int = today_result.scalar() or 0

    # Top intents
    intent_stmt = (
        select(QueryLog.intent, func.count(QueryLog.id).label("cnt"))
        .group_by(QueryLog.intent)
        .order_by(func.count(QueryLog.id).desc())
        .limit(10)
    )
    intent_result = await db.execute(intent_stmt)
    top_intents: dict[str, int] = {
        row.intent: row.cnt for row in intent_result
    }

    # Top routes (reuse popular_routes table)
    popular = await get_popular_routes(db, limit=10)
    top_routes = [
        {
            "source": r.source,
            "destination": r.destination,
            "count": r.query_count,
        }
        for r in popular
    ]

    return {
        "total_queries": total_queries,
        "queries_today": queries_today,
        "top_intents": top_intents,
        "top_routes": top_routes,
    }
