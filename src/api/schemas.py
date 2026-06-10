"""
Pydantic v2 request / response schemas for the DelhiCommuteBot API.

All models use ``model_config = ConfigDict(...)`` for strict validation
and camelCase JSON serialisation where appropriate.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared / nested models
# ---------------------------------------------------------------------------

class TransportOption(BaseModel):
    """A single transport-mode suggestion returned to the user."""

    model_config = ConfigDict(populate_by_name=True)

    mode: str = Field(
        ...,
        description="Transport mode: bus, auto, metro, shared_auto, e_rickshaw",
    )
    route_info: str = Field(
        ..., description="Human-readable route or line description",
    )
    estimated_time: str | None = Field(
        None, description="Estimated travel time, e.g. '25 min'",
    )
    fare_inr: int | None = Field(
        None, description="Estimated fare in INR",
    )
    crowding: str | None = Field(
        None,
        description="Crowding level: low / moderate / high (when available)",
    )


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Incoming commute query from a user (typically via WhatsApp)."""

    raw_text: str = Field(
        ..., min_length=1, max_length=1024,
        description="Free-text message from the user",
    )
    user_phone: str | None = Field(
        None,
        description="User's phone number (hashed before storage)",
    )
    session_id: str | None = Field(
        None,
        description="Optional session ID for multi-turn conversation context",
    )


class QueryResponse(BaseModel):
    """Full response returned after pipeline processing."""

    intent: str
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    source: str | None = None
    destination: str | None = None
    response_text: str
    options: list[TransportOption] = Field(default_factory=list)
    response_time_ms: int
    session_context: dict | None = Field(
        None,
        description="Session context carried forward for multi-turn conversations",
    )


# ---------------------------------------------------------------------------
# /classify (debug)
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    """Payload for the debug classification endpoint."""

    text: str = Field(..., min_length=1, max_length=1024)


class ClassifyResponse(BaseModel):
    """Result of intent classification only."""

    intent: str
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# /feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    """User feedback on a bot response."""

    query_log_id: int = Field(
        ..., description="ID of the query_logs entry being rated",
    )
    rating: int = Field(
        ..., ge=1, le=3,
        description="1 = thumbs down, 2 = neutral, 3 = thumbs up",
    )
    comment: str | None = Field(
        None, max_length=1024,
        description="Optional free-text feedback",
    )
    is_incorrect: bool = Field(
        False,
        description="True if the user reports the response as factually wrong",
    )
    user_phone: str | None = Field(
        None,
        description="User's phone number (hashed before storage)",
    )


class FeedbackResponse(BaseModel):
    """Acknowledgement of feedback submission."""

    status: str = "received"
    feedback_id: int
    message: str = "Thank you for your feedback! 🙏"


# ---------------------------------------------------------------------------
# /stats
# ---------------------------------------------------------------------------

class StatsResponse(BaseModel):
    """Aggregate analytics for the dashboard."""

    total_queries: int
    queries_today: int
    top_routes: list[dict] = Field(default_factory=list)
    top_intents: dict[str, int] = Field(default_factory=dict)
    feedback_summary: dict = Field(
        default_factory=dict,
        description="Aggregate feedback stats: avg_rating, total_feedback, etc.",
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Lightweight health-check response."""

    status: str = "ok"
    version: str = "0.2.0"
    uptime_seconds: float
    components: dict = Field(
        default_factory=dict,
        description="Status of sub-components: db, retriever, classifier",
    )
