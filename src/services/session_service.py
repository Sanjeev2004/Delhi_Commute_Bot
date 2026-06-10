"""
In-memory session management for DelhiCommuteBot.

Provides a thread-safe (via :class:`asyncio.Lock`) session store with
automatic TTL-based expiration.  Sessions are keyed by a hashed phone
number and carry contextual state for multi-turn conversations — enabling
follow-up queries like *"what about metro?"* to inherit source/destination
from the previous turn.

Usage::

    manager = SessionManager(ttl_minutes=30)

    await manager.update_session(
        phone_hash="abc123",
        intent="bus_route",
        source="Lajpat Nagar",
        dest="Rajiv Chowk",
        query="bus from lajpat nagar to rajiv chowk",
        response="Take DTC 505 …",
    )

    session = await manager.get_session("abc123")
    hint = await manager.get_context_hint("abc123")
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Session data container
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """In-memory representation of a user's conversation session.

    Attributes
    ----------
    last_intent:
        Most recent classified intent (e.g. ``"bus_route"``, ``"metro_route"``).
    last_source:
        Most recent source location mentioned by the user.
    last_destination:
        Most recent destination location mentioned by the user.
    last_query_time:
        Epoch timestamp of the most recent interaction.
    conversation_history:
        Rolling list of the last *N* query/response pairs (default 5).
    message_count:
        Total number of messages in this session.
    """

    last_intent: str | None = None
    last_source: str | None = None
    last_destination: str | None = None
    last_query_time: float = field(default_factory=time.time)
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    message_count: int = 0

    @property
    def is_follow_up_candidate(self) -> bool:
        """Return ``True`` if this session has enough context for follow-ups."""
        return bool(self.last_source or self.last_destination)


# ---------------------------------------------------------------------------
# Session constants
# ---------------------------------------------------------------------------

DEFAULT_TTL_MINUTES: int = 30
"""Default session time-to-live in minutes."""

MAX_CONVERSATION_HISTORY: int = 5
"""Maximum number of query/response pairs to retain per session."""

# Intent categories that share source/destination context
_TRANSPORT_INTENTS: frozenset[str] = frozenset({
    "bus_route",
    "metro_route",
    "auto_fare",
    "shared_auto",
    "route_info",
    "commute",
    "best_route",
})


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """Async-safe in-memory session store with TTL-based expiration.

    Parameters
    ----------
    ttl_minutes:
        Number of minutes before an idle session is eligible for eviction.
        Defaults to :pydata:`DEFAULT_TTL_MINUTES` (30).
    max_history:
        Maximum conversation history entries to keep per session.
        Defaults to :pydata:`MAX_CONVERSATION_HISTORY` (5).
    """

    def __init__(
        self,
        ttl_minutes: int = DEFAULT_TTL_MINUTES,
        max_history: int = MAX_CONVERSATION_HISTORY,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self._ttl_seconds: float = ttl_minutes * 60.0
        self._max_history: int = max_history
        self._lock: asyncio.Lock = asyncio.Lock()

        logger.info(
            "SessionManager initialised (TTL={}min, max_history={})",
            ttl_minutes,
            max_history,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_session(self, phone_hash: str) -> Session | None:
        """Retrieve a session by *phone_hash*, or ``None`` if expired/missing.

        Parameters
        ----------
        phone_hash:
            SHA-256 hex digest of the user's phone number.

        Returns
        -------
        Session | None
            The session object if it exists and is not expired.
        """
        async with self._lock:
            session = self._sessions.get(phone_hash)

            if session is None:
                return None

            # Check TTL
            if self._is_expired(session):
                del self._sessions[phone_hash]
                logger.debug(
                    "Session for {} expired and removed", phone_hash[:8] + "…"
                )
                return None

            return session

    async def update_session(
        self,
        phone_hash: str,
        intent: str,
        source: str | None = None,
        dest: str | None = None,
        query: str = "",
        response: str = "",
    ) -> Session:
        """Create or update a session with the latest interaction context.

        Handles follow-up logic: if the new query provides only an intent
        change (e.g. ``"metro_route"`` after ``"bus_route"``) but no new
        source/destination, the previous source/destination are carried
        over automatically.

        Parameters
        ----------
        phone_hash:
            SHA-256 hex digest of the user's phone number.
        intent:
            The classified intent of the current query.
        source:
            Extracted source location (may be ``None`` for follow-ups).
        dest:
            Extracted destination location (may be ``None`` for follow-ups).
        query:
            The raw user query text.
        response:
            The bot's response text.

        Returns
        -------
        Session
            The updated (or newly created) session.
        """
        async with self._lock:
            session = self._sessions.get(phone_hash)
            now = time.time()

            if session is None or self._is_expired(session):
                # Create fresh session
                session = Session(
                    last_intent=intent,
                    last_source=source,
                    last_destination=dest,
                    last_query_time=now,
                    message_count=1,
                )
                logger.debug(
                    "Created new session for {}", phone_hash[:8] + "…"
                )
            else:
                # Follow-up logic: carry over source/dest if not provided
                effective_source = source
                effective_dest = dest

                if self._is_transport_intent(intent) and session.is_follow_up_candidate:
                    if not effective_source and session.last_source:
                        effective_source = session.last_source
                        logger.debug(
                            "Carried over source '{}' from previous turn",
                            effective_source,
                        )
                    if not effective_dest and session.last_destination:
                        effective_dest = session.last_destination
                        logger.debug(
                            "Carried over destination '{}' from previous turn",
                            effective_dest,
                        )

                session.last_intent = intent
                session.last_source = effective_source
                session.last_destination = effective_dest
                session.last_query_time = now
                session.message_count += 1

            # Append to conversation history (bounded)
            if query or response:
                session.conversation_history.append({
                    "query": query,
                    "response": response,
                    "intent": intent,
                    "timestamp": now,
                })
                # Trim to max history
                if len(session.conversation_history) > self._max_history:
                    session.conversation_history = session.conversation_history[
                        -self._max_history :
                    ]

            self._sessions[phone_hash] = session
            return session

    async def get_context_hint(self, phone_hash: str) -> str | None:
        """Generate a human-readable context hint for follow-up queries.

        If the user has an active session with a previous source/destination,
        this returns a string like::

            "Continuing from your previous query about Lajpat Nagar → Rajiv Chowk"

        Otherwise returns ``None``.

        Parameters
        ----------
        phone_hash:
            SHA-256 hex digest of the user's phone number.

        Returns
        -------
        str | None
            Context hint string, or ``None`` if no context is available.
        """
        session = await self.get_session(phone_hash)

        if session is None or not session.is_follow_up_candidate:
            return None

        parts: list[str] = []

        if session.last_source and session.last_destination:
            parts.append(
                f"from {session.last_source} to {session.last_destination}"
            )
        elif session.last_source:
            parts.append(f"from {session.last_source}")
        elif session.last_destination:
            parts.append(f"to {session.last_destination}")

        if session.last_intent:
            intent_label = session.last_intent.replace("_", " ")
            parts.append(f"(previous intent: {intent_label})")

        if not parts:
            return None

        return "Continuing from your previous query " + " ".join(parts)

    async def clear_expired_sessions(self) -> int:
        """Remove all sessions that have exceeded their TTL.

        Returns
        -------
        int
            Number of sessions removed.
        """
        async with self._lock:
            expired_keys = [
                key
                for key, session in self._sessions.items()
                if self._is_expired(session)
            ]

            for key in expired_keys:
                del self._sessions[key]

            if expired_keys:
                logger.info(
                    "Cleared {} expired session(s) ({} remaining)",
                    len(expired_keys),
                    len(self._sessions),
                )

            return len(expired_keys)

    async def get_active_session_count(self) -> int:
        """Return the number of currently active (non-expired) sessions."""
        async with self._lock:
            return sum(
                1
                for s in self._sessions.values()
                if not self._is_expired(s)
            )

    async def delete_session(self, phone_hash: str) -> bool:
        """Explicitly delete a session.

        Returns
        -------
        bool
            ``True`` if a session was found and deleted.
        """
        async with self._lock:
            if phone_hash in self._sessions:
                del self._sessions[phone_hash]
                logger.debug(
                    "Deleted session for {}", phone_hash[:8] + "…"
                )
                return True
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_expired(self, session: Session) -> bool:
        """Check if *session* has exceeded the TTL."""
        return (time.time() - session.last_query_time) > self._ttl_seconds

    @staticmethod
    def _is_transport_intent(intent: str) -> bool:
        """Return ``True`` if *intent* is a transport query that can share context."""
        return intent.lower() in _TRANSPORT_INTENTS


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

session_manager = SessionManager()
"""Default session manager instance used across the application."""
