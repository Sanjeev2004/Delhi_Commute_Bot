"""
FastAPI router — all public-facing endpoints.

Endpoints
---------
- ``POST /query``    — Full commute-query pipeline
- ``POST /classify`` — Debug intent-classification only
- ``POST /feedback`` — Submit user feedback
- ``GET  /health``   — Health-check
- ``GET  /stats``    — Query analytics dashboard
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
    TransportOption,
)
from src.db.crud import get_stats, log_query, update_popular_routes
from src.db.database import get_db
from src.rag.chains import ResponseFormatter
from src.rag.retriever import TransitRetriever

router = APIRouter(tags=["commute"])

# ---------------------------------------------------------------------------
# Module-level singletons (populated during lifespan startup)
# ---------------------------------------------------------------------------

_retriever: TransitRetriever | None = None
_formatter = ResponseFormatter()
_start_time: float = time.time()

# Lazy imports for optional classifier / entity extractor modules.
# They may not exist yet, so guard imports and fall back to stubs.
_classifier: Any = None
_entity_extractor: Any = None
_session_manager: Any = None


def set_retriever(retriever: TransitRetriever) -> None:
    """Called from ``lifespan`` to inject the retriever singleton."""
    global _retriever
    _retriever = retriever


def set_classifier(classifier: Any) -> None:
    """Called from ``lifespan`` to inject the intent classifier."""
    global _classifier
    _classifier = classifier


def set_entity_extractor(extractor: Any) -> None:
    """Called from ``lifespan`` to inject the entity extractor."""
    global _entity_extractor
    _entity_extractor = extractor


def set_session_manager(manager: Any) -> None:
    """Called from ``lifespan`` to inject the session manager."""
    global _session_manager
    _session_manager = manager


# ---------------------------------------------------------------------------
# Stub helpers (used until real classifier / extractor modules exist)
# ---------------------------------------------------------------------------


def _classify_intent(text: str) -> tuple[str, float]:
    """Return ``(intent, confidence)`` using the loaded classifier.

    Falls back to a simple keyword-based heuristic when no trained
    classifier is available.
    """
    if _classifier is not None:
        try:
            result = _classifier.predict(text)
            return result["intent"], result["confidence"]
        except Exception as exc:
            logger.warning("Classifier failed, falling back to keywords: {}", exc)

    # ── Keyword fallback ────────────────────────────────────────────────
    lower = text.lower()

    if any(kw in lower for kw in ("compare", "vs", "versus", "options", "which is")):
        return "compare", 0.75

    if any(kw in lower for kw in ("bus", "dtc", "blueline")):
        return "bus_route", 0.80

    if any(kw in lower for kw in ("metro", "dmrc", "line")):
        return "metro_route", 0.80

    if any(kw in lower for kw in ("auto", "rickshaw", "fare")):
        return "auto_fare", 0.80

    if any(kw in lower for kw in ("shared auto", "shared", "e-rickshaw", "e rickshaw")):
        return "shared_auto", 0.80

    # Hindi/Hinglish keyword fallback
    if any(kw in lower for kw in ("बस", "bus kahan", "bus milegi")):
        return "bus_route", 0.70

    if any(kw in lower for kw in ("मेट्रो", "metro ka", "metro kaise")):
        return "metro_route", 0.70

    if any(kw in lower for kw in ("ऑटो", "auto kitne", "किराया")):
        return "auto_fare", 0.70

    if any(kw in lower for kw in ("शेयर", "shared")):
        return "shared_auto", 0.70

    if any(kw in lower for kw in ("hi", "hello", "namaste", "नमस्ते", "hii")):
        return "greeting", 0.90

    if any(kw in lower for kw in ("help", "मदद", "kya kar sakte", "what can you")):
        return "help", 0.85

    if any(kw in lower for kw in ("thanks", "धन्यवाद", "wrong", "galat", "not helpful")):
        return "feedback", 0.80

    return "general", 0.50


def _extract_entities(text: str) -> dict[str, str | None]:
    """Extract ``source`` and ``destination`` from free text.

    Falls back to a simple pattern matcher when the entity-extraction
    module is not loaded.
    """
    if _entity_extractor is not None:
        try:
            return _entity_extractor.extract(text)
        except Exception as exc:
            logger.warning("Entity extractor failed, using pattern match: {}", exc)

    # ── Pattern fallback ────────────────────────────────────────────────
    import re

    source: str | None = None
    destination: str | None = None

    # "from X to Y"
    m = re.search(r"from\s+(.+?)\s+to\s+(.+?)(?:\s*[?.!]|$)", text, re.I)
    if m:
        source = m.group(1).strip().title()
        destination = m.group(2).strip().title()
        return {"source": source, "destination": destination}

    # Hindi: "X से Y तक" or "X se Y tak"
    m = re.search(r"(.+?)\s+(?:से|se)\s+(.+?)(?:\s+(?:तक|tak|को|ko))?(?:\s*[?.!]|$)", text, re.I)
    if m:
        source = m.group(1).strip().title()
        destination = m.group(2).strip().title()
        return {"source": source, "destination": destination}

    # "X to Y"
    m = re.search(r"^(.+?)\s+to\s+(.+?)(?:\s*[?.!]|$)", text, re.I)
    if m:
        source = m.group(1).strip().title()
        destination = m.group(2).strip().title()

    # "near X" / "around X"
    m = re.search(r"(?:near|around|at)\s+(.+?)(?:\s*[?.!]|$)", text, re.I)
    if m and not source:
        source = m.group(1).strip().title()

    return {"source": source, "destination": destination}


# ---------------------------------------------------------------------------
# Session context helpers
# ---------------------------------------------------------------------------


def _get_session_context(user_phone: str | None) -> dict | None:
    """Retrieve conversation context for follow-up queries."""
    if _session_manager is None or not user_phone:
        return None

    import hashlib
    phone_hash = hashlib.sha256(user_phone.encode()).hexdigest()

    try:
        session = _session_manager.get_session(phone_hash)
        if session:
            return {
                "last_intent": session.get("last_intent"),
                "last_source": session.get("last_source"),
                "last_destination": session.get("last_destination"),
            }
    except Exception as exc:
        logger.warning("Failed to get session context: {}", exc)

    return None


def _update_session(
    user_phone: str | None,
    intent: str,
    source: str | None,
    dest: str | None,
    query: str,
    response: str,
) -> None:
    """Update conversation session after a query."""
    if _session_manager is None or not user_phone:
        return

    import hashlib
    phone_hash = hashlib.sha256(user_phone.encode()).hexdigest()

    try:
        _session_manager.update_session(
            phone_hash,
            intent=intent,
            source=source,
            destination=dest,
            query=query,
            response=response,
        )
    except Exception as exc:
        logger.warning("Failed to update session: {}", exc)


def _apply_session_context(
    entities: dict[str, str | None],
    session_ctx: dict | None,
) -> dict[str, str | None]:
    """Fill in missing source/destination from previous conversation turn."""
    if not session_ctx:
        return entities

    if not entities.get("source") and session_ctx.get("last_source"):
        entities["source"] = session_ctx["last_source"]
        logger.debug("Inherited source from session: {}", entities["source"])

    if not entities.get("destination") and session_ctx.get("last_destination"):
        entities["destination"] = session_ctx["last_destination"]
        logger.debug("Inherited destination from session: {}", entities["destination"])

    return entities


# ---------------------------------------------------------------------------
# Helpers for building response from RAG docs
# ---------------------------------------------------------------------------


def _docs_to_bus_routes(docs: list) -> list[dict[str, Any]]:
    """Convert RAG bus documents into route dicts for the formatter."""
    routes: list[dict[str, Any]] = []
    for doc in docs:
        meta = doc.metadata
        if meta.get("type") != "bus":
            continue
        routes.append({
            "route_id": meta.get("route_id", "?"),
            "name": meta.get("name", ""),
            "via": "",
            "fare_inr": 15,  # Default DTC fare
            "crowding": "moderate",
            "estimated_time": None,
        })
    return routes


def _docs_to_metro_path(docs: list) -> dict[str, Any]:
    """Build a simplified metro path dict from RAG metro documents."""
    for doc in docs:
        meta = doc.metadata
        if meta.get("type") in ("metro", "metro_station"):
            return {
                "line": meta.get("line", "?"),
                "stations": meta.get("station_count", "?"),
                "interchanges": [],
                "fare_inr": 22,  # Default slab
                "estimated_time": "~20 min",
                "walking_time": "5 min",
            }
    return {
        "line": "Unknown",
        "stations": "?",
        "interchanges": [],
        "fare_inr": None,
        "estimated_time": None,
        "walking_time": None,
    }


def _docs_to_auto_fare(docs: list) -> dict[str, Any]:
    """Extract auto fare info from RAG documents."""
    for doc in docs:
        meta = doc.metadata
        if meta.get("type") == "auto_fare" and meta.get("subtype") == "route":
            return {
                "meter_fare_inr": meta.get("meter_fare_inr", 85),
                "asking_fare_inr": meta.get("asking_fare_inr"),
                "distance_km": meta.get("distance_km"),
                "estimated_time": None,
            }
    # Default calculation
    return {
        "meter_fare_inr": 85,
        "asking_fare_inr": 120,
        "distance_km": None,
        "estimated_time": "~25 min",
    }


def _docs_to_shared_routes(docs: list) -> list[dict[str, Any]]:
    """Convert RAG shared-auto documents into route dicts."""
    routes: list[dict[str, Any]] = []
    for doc in docs:
        meta = doc.metadata
        if meta.get("type") != "shared_auto":
            continue
        routes.append({
            "route_id": meta.get("route_id", "?"),
            "name": doc.page_content.split(":")[0] if ":" in doc.page_content else "",
            "from": meta.get("from", ""),
            "to": meta.get("to", ""),
            "fare_inr": meta.get("fare_inr"),
            "type": "shared_auto",
            "frequency": "",
            "operating_hours": "",
        })
    return routes


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/query", response_model=QueryResponse)
async def query_commute(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """Main commute-query pipeline.

    1. Retrieve session context (if available)
    2. Classify intent
    3. Extract entities (source / destination)
    4. Apply session context for follow-up queries
    5. Retrieve relevant docs via RAG
    6. Format WhatsApp-friendly response
    7. Log to DB & update session
    8. Return structured response
    """
    start = time.perf_counter()

    try:
        # 1. Session context
        session_ctx = _get_session_context(request.user_phone)

        # 2. Classify
        intent, confidence = _classify_intent(request.raw_text)

        # 3. Extract entities
        entities = _extract_entities(request.raw_text)
        source = entities.get("source")
        destination = entities.get("destination")

        # 4. Apply session context for follow-up queries
        if not source or not destination:
            entities = _apply_session_context(entities, session_ctx)
            source = entities.get("source")
            destination = entities.get("destination")

        # 5. Retrieve via RAG
        response_text: str
        options: list[TransportOption] = []

        if _retriever is not None:
            docs = _retriever.retrieve(
                request.raw_text, intent=intent, k=5,
            )
        else:
            docs = []
            logger.warning("No retriever loaded — returning fallback response")

        # 6. Format response based on intent
        match intent:
            case "bus_route":
                bus_routes = _docs_to_bus_routes(docs)
                response_text, options = _formatter.format_bus_response(
                    bus_routes, source or "?", destination or "?",
                )
            case "metro_route":
                metro_path = _docs_to_metro_path(docs)
                response_text, options = _formatter.format_metro_response(
                    metro_path, source or "?", destination or "?",
                )
            case "auto_fare":
                fare_info = _docs_to_auto_fare(docs)
                response_text, options = _formatter.format_auto_response(
                    fare_info, source or "?", destination or "?",
                )
            case "shared_auto":
                shared_routes = _docs_to_shared_routes(docs)
                response_text, options = _formatter.format_shared_auto_response(
                    shared_routes,
                )
            case "compare":
                compare_options: list[dict[str, Any]] = []
                bus_routes = _docs_to_bus_routes(docs)
                if bus_routes:
                    r = bus_routes[0]
                    compare_options.append({
                        "mode": "bus",
                        "route_info": f"Route {r['route_id']}",
                        "fare_inr": r.get("fare_inr"),
                        "estimated_time": "~30 min",
                    })
                metro_path = _docs_to_metro_path(docs)
                if metro_path.get("line") != "Unknown":
                    compare_options.append({
                        "mode": "metro",
                        "route_info": f"{metro_path['line']}, {metro_path['stations']} stops",
                        "fare_inr": metro_path.get("fare_inr"),
                        "estimated_time": metro_path.get("estimated_time"),
                    })
                fare_info = _docs_to_auto_fare(docs)
                compare_options.append({
                    "mode": "auto",
                    "route_info": f"{source or '?'} → {destination or '?'}",
                    "fare_inr": fare_info.get("meter_fare_inr"),
                    "estimated_time": fare_info.get("estimated_time", "~25 min"),
                })
                response_text, options = _formatter.format_compare_response(
                    compare_options, source or "?", destination or "?",
                )
            case "greeting":
                response_text = _formatter.format_greeting_response()
            case "help":
                response_text = _formatter.format_help_response()
            case "feedback":
                response_text = _formatter.format_feedback_ack_response()
            case _:
                response_text = _formatter.format_fallback_response(
                    request.raw_text,
                )

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        # 7. Log to DB & update session
        try:
            await log_query(
                db,
                raw_query=request.raw_text,
                intent=intent,
                source_location=source,
                destination_location=destination,
                response_text=response_text,
                response_time_ms=elapsed_ms,
                user_phone=request.user_phone,
            )
            if source and destination:
                await update_popular_routes(db, source, destination)
        except Exception as db_exc:
            logger.error("Failed to log query to DB: {}", db_exc)

        # Update session for multi-turn context
        _update_session(
            request.user_phone, intent, source, destination,
            request.raw_text, response_text,
        )

        # 8. Return
        return QueryResponse(
            intent=intent,
            confidence=confidence,
            source=source,
            destination=destination,
            response_text=response_text,
            options=options,
            response_time_ms=elapsed_ms,
            session_context=session_ctx,
        )

    except Exception as exc:
        logger.exception("Unhandled error in /query")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/classify", response_model=ClassifyResponse)
async def classify_text(request: ClassifyRequest) -> ClassifyResponse:
    """Debug endpoint — returns only intent classification."""
    try:
        intent, confidence = _classify_intent(request.text)
        return ClassifyResponse(intent=intent, confidence=confidence)
    except Exception as exc:
        logger.exception("Error in /classify")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    """Submit user feedback on a bot response."""
    import hashlib
    from src.db.models import UserFeedback

    try:
        phone_hash = (
            hashlib.sha256(request.user_phone.encode()).hexdigest()
            if request.user_phone
            else "anonymous"
        )

        feedback = UserFeedback(
            query_log_id=request.query_log_id,
            user_phone_hash=phone_hash,
            rating=request.rating,
            comment=request.comment,
            is_incorrect=request.is_incorrect,
        )
        db.add(feedback)
        await db.flush()

        logger.info(
            "Feedback submitted: id={} query={} rating={}",
            feedback.id, request.query_log_id, request.rating,
        )

        return FeedbackResponse(
            feedback_id=feedback.id,
            message=(
                "Thank you for your feedback! 🙏"
                if request.rating >= 2
                else "Sorry about that! We'll work on improving. 🙏"
            ),
        )

    except Exception as exc:
        logger.exception("Error in /feedback")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Lightweight health check with component status."""
    components = {
        "retriever": "loaded" if _retriever is not None else "not_loaded",
        "classifier": "loaded" if _classifier is not None else "keyword_fallback",
        "entity_extractor": "loaded" if _entity_extractor is not None else "pattern_fallback",
        "session_manager": "loaded" if _session_manager is not None else "disabled",
    }
    return HealthResponse(
        status="ok",
        version="0.2.0",
        uptime_seconds=round(time.time() - _start_time, 2),
        components=components,
    )


@router.get("/stats", response_model=StatsResponse)
async def stats_dashboard(
    db: AsyncSession = Depends(get_db),
) -> StatsResponse:
    """Return aggregate query analytics including feedback."""
    try:
        data = await get_stats(db)
        return StatsResponse(**data)
    except Exception as exc:
        logger.exception("Error in /stats")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
