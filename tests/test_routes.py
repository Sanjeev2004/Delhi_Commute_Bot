"""
Tests for the FastAPI API endpoints (src.api.routes).

Uses ``httpx.AsyncClient`` with the FastAPI ASGI transport so no
real HTTP server is started.  The database dependency is overridden
by conftest's ``app_client`` fixture (in-memory SQLite).

Covered endpoints:
- ``GET  /health``
- ``POST /query``   (bus, metro, missing text)
- ``POST /classify``
- ``GET  /stats``
- DB logging verification
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import QueryLog


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint(app_client: AsyncClient) -> None:
    """GET /health should return 200 with status='ok'."""
    response = await app_client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "ok", f"Expected status='ok', got '{data['status']}'"
    assert "version" in data, "Response should include 'version'"
    assert "uptime_seconds" in data, "Response should include 'uptime_seconds'"


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_endpoint_bus(app_client: AsyncClient) -> None:
    """POST /query with a bus query should return intent='bus_route'."""
    payload = {
        "raw_text": "bus from Kashmere Gate to Laxmi Nagar",
        "user_phone": "+919876543210",
    }
    response = await app_client.post("/query", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["intent"] == "bus_route", (
        f"Expected intent='bus_route', got '{data['intent']}'"
    )
    assert "response_text" in data, "Response should include 'response_text'"
    assert "response_time_ms" in data, "Response should include 'response_time_ms'"


@pytest.mark.asyncio
async def test_query_endpoint_metro(app_client: AsyncClient) -> None:
    """POST /query with a metro query should return intent='metro_route'."""
    payload = {
        "raw_text": "metro route to Rajiv Chowk",
        "user_phone": "+919876543210",
    }
    response = await app_client.post("/query", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["intent"] == "metro_route", (
        f"Expected intent='metro_route', got '{data['intent']}'"
    )


@pytest.mark.asyncio
async def test_query_endpoint_missing_text(app_client: AsyncClient) -> None:
    """POST /query with empty raw_text should return 422 validation error."""
    payload = {"raw_text": "", "user_phone": "+911234567890"}
    response = await app_client.post("/query", json=payload)
    assert response.status_code == 422, (
        f"Expected 422 for empty text, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# Classify endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_endpoint(app_client: AsyncClient) -> None:
    """POST /classify should return intent and confidence."""
    payload = {"text": "bus from CP to AIIMS"}
    response = await app_client.post("/classify", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "intent" in data, "Response should include 'intent'"
    assert "confidence" in data, "Response should include 'confidence'"
    assert isinstance(data["confidence"], (int, float)), (
        f"Confidence should be numeric, got {type(data['confidence'])}"
    )
    assert 0.0 <= data["confidence"] <= 1.0, (
        f"Confidence should be between 0 and 1, got {data['confidence']}"
    )


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stats_endpoint(app_client: AsyncClient) -> None:
    """GET /stats should return aggregate analytics data."""
    response = await app_client.get("/stats")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "total_queries" in data, "Response should include 'total_queries'"
    assert "queries_today" in data, "Response should include 'queries_today'"
    assert isinstance(data["total_queries"], int), (
        f"total_queries should be int, got {type(data['total_queries'])}"
    )
    assert isinstance(data["queries_today"], int), (
        f"queries_today should be int, got {type(data['queries_today'])}"
    )


# ---------------------------------------------------------------------------
# DB logging after query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_logs_to_db(
    app_client: AsyncClient,
    async_db_session: AsyncSession,
) -> None:
    """After POST /query, a QueryLog row should exist in the database."""
    payload = {
        "raw_text": "auto fare from CP to Saket",
        "user_phone": "+911111111111",
    }
    response = await app_client.post("/query", json=payload)
    assert response.status_code == 200, f"Query failed: {response.status_code}"

    # Verify the log was written
    result = await async_db_session.execute(
        select(QueryLog).where(QueryLog.raw_query == "auto fare from CP to Saket")
    )
    log_entry = result.scalar_one_or_none()
    assert log_entry is not None, (
        "Expected a QueryLog row for the submitted query, but none was found"
    )
    assert log_entry.intent is not None, "QueryLog.intent should be populated"
    assert log_entry.response_text, "QueryLog.response_text should not be empty"
    assert log_entry.response_time_ms >= 0, (
        f"response_time_ms should be non-negative, got {log_entry.response_time_ms}"
    )
