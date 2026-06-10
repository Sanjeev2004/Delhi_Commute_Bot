"""
FastAPI application entrypoint for DelhiCommuteBot.

Responsibilities:
- Application lifespan management (startup / shutdown hooks)
- Middleware configuration
- Router registration
- Uvicorn runner for direct execution
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    On startup:
    1. Initialise the database (create tables).
    2. Load the intent classifier (if available).
    3. Load the entity extractor (if available).
    4. Load the RAG retriever (FAISS indices).

    On shutdown:
    - Dispose of the database engine connection pool.
    """
    logger.info("Starting DelhiCommuteBot API (env={})", settings.app_env)

    # ── 1. Database ─────────────────────────────────────────────────────
    from src.db.database import init_db, engine

    await init_db()

    # ── 2. Intent classifier (optional) ─────────────────────────────────
    from src.api.routes import set_classifier

    try:
        from pathlib import Path

        model_path = Path(settings.intent_model_path)
        if model_path.exists():
            from src.classifier.intent import IntentClassifier  # type: ignore[import-untyped]

            classifier = IntentClassifier.load(str(model_path))
            set_classifier(classifier)
            logger.info("Intent classifier loaded from {}", model_path)
        else:
            logger.warning(
                "Intent classifier model not found at {} — using keyword fallback",
                model_path,
            )
    except ImportError:
        logger.warning("Classifier module not available — using keyword fallback")
    except Exception as exc:
        logger.warning("Failed to load classifier: {} — using keyword fallback", exc)

    # ── 3. Entity extractor (optional) ──────────────────────────────────
    from src.api.routes import set_entity_extractor

    try:
        from src.classifier.entities import EntityExtractor  # type: ignore[import-untyped]

        extractor = EntityExtractor()
        set_entity_extractor(extractor)
        logger.info("Entity extractor loaded")
    except ImportError:
        logger.warning("Entity extractor module not available — using pattern fallback")
    except Exception as exc:
        logger.warning("Failed to load entity extractor: {} — using pattern fallback", exc)

    # ── 4. RAG retriever ────────────────────────────────────────────────
    from src.api.routes import set_retriever

    try:
        from src.rag.retriever import TransitRetriever
        from pathlib import Path

        index_dir = Path(settings.faiss_index_path)
        if index_dir.exists() and any(index_dir.iterdir()):
            retriever = TransitRetriever()
            set_retriever(retriever)
            logger.info(
                "RAG retriever loaded (indices: {})",
                retriever.available_indices,
            )
        else:
            logger.warning(
                "No FAISS indices found at {}. Run "
                "`python -m src.rag.indexer` to build them.",
                index_dir,
            )
    except Exception as exc:
        logger.warning("Failed to load RAG retriever: {}", exc)

    logger.info("Startup complete ✅")

    yield

    # ── Shutdown ────────────────────────────────────────────────────────
    logger.info("Shutting down …")
    await engine.dispose()
    logger.info("Shutdown complete 🛑")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DelhiCommuteBot API",
    description=(
        "WhatsApp-based Delhi commute assistant — "
        "bus, metro, auto, and shared-auto route information."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS (permissive for development) ───────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ───────────────────────────────────────────────────────

from src.api.routes import router as api_router  # noqa: E402

app.include_router(api_router)


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )
