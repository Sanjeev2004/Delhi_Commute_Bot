"""
Feedback service for DelhiCommuteBot.

Provides async methods to submit and aggregate user feedback on bot
responses.  Feedback is stored in the :class:`~src.db.models.UserFeedback`
table and linked to :class:`~src.db.models.QueryLog` entries via a soft
foreign key.

Usage::

    from src.services.feedback_service import feedback_service

    # Submit feedback
    fb = await feedback_service.submit_feedback(
        db=session,
        query_log_id=42,
        rating=3,
        comment="Very helpful!",
    )

    # Retrieve aggregate stats
    stats = await feedback_service.get_feedback_stats(db=session)
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.db.models import UserFeedback


# ---------------------------------------------------------------------------
# Rating constants
# ---------------------------------------------------------------------------

RATING_THUMBS_DOWN: int = 1
RATING_NEUTRAL: int = 2
RATING_THUMBS_UP: int = 3

_VALID_RATINGS: frozenset[int] = frozenset({
    RATING_THUMBS_DOWN,
    RATING_NEUTRAL,
    RATING_THUMBS_UP,
})


# ---------------------------------------------------------------------------
# FeedbackService
# ---------------------------------------------------------------------------

class FeedbackService:
    """Async service for user feedback CRUD and analytics.

    All public methods accept an ``AsyncSession`` as the first argument
    so they integrate naturally with FastAPI's ``Depends(get_db)``
    dependency injection.
    """

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def submit_feedback(
        self,
        db: AsyncSession,
        query_log_id: int,
        rating: int,
        comment: str | None = None,
        *,
        user_phone_hash: str = "anonymous",
        is_incorrect: bool = False,
    ) -> UserFeedback:
        """Persist a new feedback entry in the database.

        Parameters
        ----------
        db:
            Active async database session.
        query_log_id:
            The ``QueryLog.id`` that this feedback refers to.
        rating:
            Feedback rating — ``1`` (thumbs down), ``2`` (neutral), or
            ``3`` (thumbs up).
        comment:
            Optional free-text comment from the user.
        user_phone_hash:
            SHA-256 hash of the user's phone number (default ``"anonymous"``).
        is_incorrect:
            ``True`` if the user flagged the bot response as incorrect.

        Returns
        -------
        UserFeedback
            The persisted feedback row (with ``id`` populated after flush).

        Raises
        ------
        ValueError
            If *rating* is not one of {1, 2, 3}.
        """
        if rating not in _VALID_RATINGS:
            raise ValueError(
                f"Invalid rating {rating!r}. Must be one of {sorted(_VALID_RATINGS)}"
            )

        feedback = UserFeedback(
            query_log_id=query_log_id,
            user_phone_hash=user_phone_hash,
            rating=rating,
            comment=comment.strip() if comment else None,
            is_incorrect=is_incorrect,
        )
        db.add(feedback)
        await db.flush()

        logger.info(
            "Feedback submitted (id={}, query_log={}, rating={}, incorrect={})",
            feedback.id,
            query_log_id,
            rating,
            is_incorrect,
        )
        return feedback

    # ------------------------------------------------------------------
    # Read / analytics operations
    # ------------------------------------------------------------------

    async def get_feedback_stats(self, db: AsyncSession) -> dict[str, Any]:
        """Return aggregate feedback statistics.

        Returns
        -------
        dict
            Keys:

            - ``total_feedback`` — total number of feedback entries.
            - ``average_rating`` — mean rating (float, rounded to 2 dp).
            - ``rating_distribution`` — dict mapping rating value → count.
            - ``thumbs_up_count`` — number of positive ratings (3).
            - ``thumbs_down_count`` — number of negative ratings (1).
            - ``neutral_count`` — number of neutral ratings (2).
            - ``incorrect_reports`` — number of times users flagged incorrect
              responses.
            - ``comments_count`` — number of entries with a non-null comment.
            - ``satisfaction_pct`` — percentage of ratings ≥ 2
              (neutral + positive).
        """
        # Total count
        total_result = await db.execute(
            select(func.count(UserFeedback.id))
        )
        total: int = total_result.scalar() or 0

        if total == 0:
            return self._empty_stats()

        # Average rating
        avg_result = await db.execute(
            select(func.avg(UserFeedback.rating))
        )
        avg_raw = avg_result.scalar()
        average_rating: float = round(float(avg_raw), 2) if avg_raw else 0.0

        # Rating distribution
        dist_stmt = (
            select(
                UserFeedback.rating,
                func.count(UserFeedback.id).label("cnt"),
            )
            .group_by(UserFeedback.rating)
        )
        dist_result = await db.execute(dist_stmt)
        distribution: dict[int, int] = {
            row.rating: row.cnt for row in dist_result
        }

        thumbs_up = distribution.get(RATING_THUMBS_UP, 0)
        thumbs_down = distribution.get(RATING_THUMBS_DOWN, 0)
        neutral = distribution.get(RATING_NEUTRAL, 0)

        # Incorrect report count
        incorrect_result = await db.execute(
            select(func.count(UserFeedback.id)).where(
                UserFeedback.is_incorrect.is_(True)
            )
        )
        incorrect_reports: int = incorrect_result.scalar() or 0

        # Comments count
        comments_result = await db.execute(
            select(func.count(UserFeedback.id)).where(
                UserFeedback.comment.isnot(None),
                UserFeedback.comment != "",
            )
        )
        comments_count: int = comments_result.scalar() or 0

        # Satisfaction percentage (neutral + thumbs up)
        satisfied = thumbs_up + neutral
        satisfaction_pct = round((satisfied / total) * 100, 1) if total else 0.0

        return {
            "total_feedback": total,
            "average_rating": average_rating,
            "rating_distribution": distribution,
            "thumbs_up_count": thumbs_up,
            "thumbs_down_count": thumbs_down,
            "neutral_count": neutral,
            "incorrect_reports": incorrect_reports,
            "comments_count": comments_count,
            "satisfaction_pct": satisfaction_pct,
        }

    async def get_recent_feedback(
        self,
        db: AsyncSession,
        limit: int = 20,
    ) -> list[UserFeedback]:
        """Return the most recent *limit* feedback entries.

        Parameters
        ----------
        db:
            Active async database session.
        limit:
            Maximum number of entries to return.

        Returns
        -------
        list[UserFeedback]
            Ordered by ``created_at`` descending.
        """
        stmt = (
            select(UserFeedback)
            .order_by(UserFeedback.created_at.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_feedback_for_query(
        self,
        db: AsyncSession,
        query_log_id: int,
    ) -> list[UserFeedback]:
        """Return all feedback entries for a specific query log.

        Parameters
        ----------
        db:
            Active async database session.
        query_log_id:
            The ``QueryLog.id`` to look up.

        Returns
        -------
        list[UserFeedback]
            All feedback entries linked to the given query.
        """
        stmt = (
            select(UserFeedback)
            .where(UserFeedback.query_log_id == query_log_id)
            .order_by(UserFeedback.created_at.desc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        """Return a stats dict with zero values (no feedback yet)."""
        return {
            "total_feedback": 0,
            "average_rating": 0.0,
            "rating_distribution": {},
            "thumbs_up_count": 0,
            "thumbs_down_count": 0,
            "neutral_count": 0,
            "incorrect_reports": 0,
            "comments_count": 0,
            "satisfaction_pct": 0.0,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

feedback_service = FeedbackService()
"""Default feedback service instance used across the application."""
