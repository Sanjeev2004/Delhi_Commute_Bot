"""
SQLAlchemy ORM models for DelhiCommuteBot.

Tables
------
- **QueryLog** – immutable audit log of every incoming user query.
- **PopularRoute** – denormalised counter tracking the most-queried
  source → destination pairs.
- **UserFeedback** – user-submitted feedback on bot responses.
- **ConversationSession** – persistent conversation context for
  multi-turn interactions.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import String, Integer, DateTime, Text, Boolean, Float, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all models."""


class QueryLog(Base):
    """Immutable log entry for each incoming commute query."""

    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    user_phone_hash: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True,
        comment="SHA-256 hash of the user phone for privacy",
    )
    raw_query: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_location: Mapped[str] = mapped_column(String(256), nullable=True)
    destination_location: Mapped[str] = mapped_column(String(256), nullable=True)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<QueryLog id={self.id} intent={self.intent!r} "
            f"src={self.source_location!r} dest={self.destination_location!r}>"
        )


class PopularRoute(Base):
    """Denormalised counter for the most-queried source → destination pairs."""

    __tablename__ = "popular_routes"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    source: Mapped[str] = mapped_column(
        String(256), nullable=False, index=True,
    )
    destination: Mapped[str] = mapped_column(
        String(256), nullable=False, index=True,
    )
    query_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1,
    )
    last_queried: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<PopularRoute {self.source!r} → {self.destination!r} "
            f"count={self.query_count}>"
        )


class UserFeedback(Base):
    """User-submitted feedback on a bot response.

    Linked to a ``QueryLog`` entry via ``query_log_id`` (soft reference,
    no FK constraint for flexibility).
    """

    __tablename__ = "user_feedback"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    query_log_id: Mapped[int] = mapped_column(
        Integer, nullable=False, index=True,
        comment="References query_logs.id (soft FK)",
    )
    user_phone_hash: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True,
    )
    rating: Mapped[int] = mapped_column(
        Integer, nullable=False,
        comment="1 = thumbs down, 2 = neutral, 3 = thumbs up",
    )
    comment: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Optional free-text feedback from the user",
    )
    is_incorrect: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
        comment="True if user reported the response as incorrect",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<UserFeedback id={self.id} query={self.query_log_id} "
            f"rating={self.rating}>"
        )


class ConversationSession(Base):
    """Persistent conversation context for multi-turn interactions.

    Each record tracks the last known state for a user so that
    follow-up queries (e.g. "what about metro?") can inherit the
    source/destination from a previous message.
    """

    __tablename__ = "conversation_sessions"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    user_phone_hash: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True,
    )
    last_intent: Mapped[str | None] = mapped_column(
        String(64), nullable=True,
    )
    last_source: Mapped[str | None] = mapped_column(
        String(256), nullable=True,
    )
    last_destination: Mapped[str | None] = mapped_column(
        String(256), nullable=True,
    )
    last_query: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    last_response: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    message_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    last_active: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<ConversationSession user={self.user_phone_hash[:8]}… "
            f"last_intent={self.last_intent!r}>"
        )
