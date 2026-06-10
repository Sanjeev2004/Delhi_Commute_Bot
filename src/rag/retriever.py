"""
Transit data retriever — loads pre-built FAISS indices and performs
similarity search against incoming queries.

Usage::

    retriever = TransitRetriever()
    docs = retriever.retrieve("bus from Kashmere Gate to Laxmi Nagar", intent="bus_route")
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from loguru import logger

from src.config import settings


_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Mapping from intent → index name(s) to search
_INTENT_INDEX_MAP: dict[str, list[str]] = {
    "bus_route": ["bus"],
    "metro_route": ["metro"],
    "auto_fare": ["auto"],
    "shared_auto": ["shared_auto"],
    "compare": ["bus", "metro", "auto", "shared_auto"],
    "general": ["bus", "metro", "auto", "shared_auto"],
}


class TransitRetriever:
    """Loads pre-built FAISS indices and retrieves relevant documents."""

    def __init__(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
        self._index_dir = Path(settings.faiss_index_path)
        self._stores: dict[str, FAISS] = {}
        self._load_indices()

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------

    def _load_indices(self) -> None:
        """Load all available FAISS indices from ``data/indices/``."""
        index_names = ["bus_index", "metro_index", "shared_auto_index", "auto_index"]

        for name in index_names:
            path = self._index_dir / name
            if path.exists():
                try:
                    store = FAISS.load_local(
                        str(path),
                        self._embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    # Derive key: "bus_index" → "bus"
                    key = name.replace("_index", "")
                    self._stores[key] = store
                    logger.info("Loaded FAISS index '{}' from {}", key, path)
                except Exception as exc:
                    logger.error("Failed to load index '{}': {}", name, exc)
            else:
                logger.warning(
                    "FAISS index '{}' not found at {} — skipping",
                    name,
                    path,
                )

        if not self._stores:
            logger.warning(
                "No FAISS indices loaded. Run `TransitIndexer.build_all()` first."
            )

    @property
    def available_indices(self) -> list[str]:
        """Return the names of currently loaded indices."""
        return list(self._stores.keys())

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        intent: str = "general",
        k: int = 5,
    ) -> list[Document]:
        """Retrieve up to *k* relevant documents based on *intent*.

        Parameters
        ----------
        query:
            Free-text search query from the user.
        intent:
            Classified intent (e.g. ``bus_route``, ``metro_route``).
            Determines which index(es) to search.
        k:
            Maximum number of documents to return per index.

        Returns
        -------
        list[Document]
            Ranked list of relevant route / station documents.
        """
        index_keys = _INTENT_INDEX_MAP.get(intent, ["bus", "metro"])
        all_docs: list[Document] = []

        for key in index_keys:
            store = self._stores.get(key)
            if store is None:
                logger.debug("Index '{}' not loaded — skipping", key)
                continue

            try:
                results = store.similarity_search(query, k=k)
                all_docs.extend(results)
                logger.debug(
                    "Retrieved {} docs from '{}' index for query={!r}",
                    len(results),
                    key,
                    query[:60],
                )
            except Exception as exc:
                logger.error(
                    "Similarity search failed on '{}': {}", key, exc,
                )

        return all_docs

    def retrieve_with_scores(
        self,
        query: str,
        intent: str = "general",
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        """Like :meth:`retrieve` but also returns similarity scores."""
        index_keys = _INTENT_INDEX_MAP.get(intent, ["bus", "metro"])
        all_results: list[tuple[Document, float]] = []

        for key in index_keys:
            store = self._stores.get(key)
            if store is None:
                continue

            try:
                results = store.similarity_search_with_score(query, k=k)
                all_results.extend(results)
            except Exception as exc:
                logger.error(
                    "Scored search failed on '{}': {}", key, exc,
                )

        # Sort by score ascending (lower = more similar in FAISS L2)
        all_results.sort(key=lambda x: x[1])
        return all_results[:k]
