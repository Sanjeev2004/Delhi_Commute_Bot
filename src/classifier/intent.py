"""
Intent classifier for DelhiCommuteBot using sentence-transformer embeddings.

Uses the all-MiniLM-L6-v2 model to embed user queries and classify them
into predefined intents via cosine similarity against pre-computed example
embeddings.

Usage:
    from src.classifier.intent import IntentClassifier

    clf = IntentClassifier()
    result = clf.predict("How do I go from CP to AIIMS by metro?")
    # {"intent": "metro_route", "confidence": 0.89}
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful fallback when sentence-transformers is unavailable
# ---------------------------------------------------------------------------
_HAS_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    logger.warning(
        "sentence-transformers is not installed. "
        "IntentClassifier will use keyword-based fallback."
    )

# ---------------------------------------------------------------------------
# Default intent→example-sentences mapping
# ---------------------------------------------------------------------------
_DEFAULT_INTENT_EXAMPLES: Dict[str, List[str]] = {
    "bus_route": [
        "bus from X to Y",
        "DTC route to Y",
        "कौन सी बस जाएगी",
        "bus kahan se milegi",
        "Which bus goes to AIIMS?",
        "bus number for Dwarka to CP",
        "Is there a direct bus from Anand Vihar?",
        "DTC bus route batao",
        "बस कौन सी लगेगी",
        "cluster bus from GTB Nagar",
        "AC bus hai kya CP se dwarka ke liye",
        "Mujhe bus chahiye sarai kale khan se",
    ],
    "metro_route": [
        "metro to X",
        "DMRC route",
        "मेट्रो कैसे जाऊं",
        "metro ka rasta batao",
        "Which metro line goes to Hauz Khas?",
        "How to reach HUDA City Centre by metro?",
        "metro interchange kahan karna padega",
        "metro mein kitna time lagega",
        "DMRC fare from Rajouri Garden to Saket",
        "nearest metro station",
        "yellow line metro stops",
        "blue line metro route",
    ],
    "auto_fare": [
        "auto fare from X to Y",
        "rickshaw cost",
        "ऑटो का किराया",
        "auto kitne ka padega",
        "How much will an auto charge?",
        "auto ka meter rate kya hai",
        "minimum auto fare in Delhi",
        "रिक्शा का किराया कितना होगा",
        "auto se jaana hai kitna lagega",
        "auto wala kitna lega",
        "3 wheeler ka rate batao",
        "What should I pay for auto?",
    ],
    "shared_auto": [
        "shared auto near X",
        "e-rickshaw",
        "शेयर ऑटो",
        "shared auto milega kya",
        "Shared auto route near Kashmere Gate",
        "Where can I find a shared auto?",
        "e-rickshaw ka rate kya hai",
        "ई-रिक्शा कहाँ मिलेगा",
        "Gramin seva route",
        "vikram auto milega kya",
        "shared auto stand kahan hai",
        "shared auto frequency",
    ],
    "compare": [
        "compare options",
        "which is cheaper",
        "कौन सा सस्ता है",
        "best way to go",
        "Compare bus and metro",
        "Should I take bus or metro?",
        "cheapest option batao",
        "metro ya bus kya better hai",
        "Auto vs metro cost comparison",
        "fastest route",
        "Time comparison between bus and metro",
        "सबसे सस्ता तरीका बताओ",
    ],
    "greeting": [
        "hi",
        "hello",
        "namaste",
        "नमस्ते",
        "hey",
        "good morning",
        "good evening",
        "hello ji",
        "kaise ho",
        "कैसे हो",
    ],
    "help": [
        "help",
        "what can you do",
        "मदद",
        "kya kar sakte ho",
        "Tell me your features",
        "How does this bot work?",
        "show me available commands",
        "मुझे मदद चाहिए",
        "guide me",
        "bot kaise use kare",
    ],
    "feedback": [
        "thanks",
        "wrong answer",
        "not helpful",
        "धन्यवाद",
        "shukriya",
        "great answer",
        "this was helpful",
        "ye galat hai",
        "this is incorrect",
        "ये गलत जवाब है",
        "bahut accha",
        "perfect answer",
    ],
    "general": [
        "What is the weather today?",
        "Tell me a joke",
        "Who are you?",
        "random question",
        "Tell me about yourself",
        "kuch bhi batao",
        "aaj kya din hai",
        "sing a song",
    ],
}

# Keywords used when sentence-transformers is not available
_KEYWORD_FALLBACK: Dict[str, List[str]] = {
    "bus_route": ["bus", "dtc", "cluster", "बस"],
    "metro_route": ["metro", "dmrc", "मेट्रो", "subway"],
    "auto_fare": ["auto", "rickshaw", "ऑटो", "रिक्शा", "fare", "cost", "charge", "किराया"],
    "shared_auto": ["shared", "e-rickshaw", "ई-रिक्शा", "शेयर", "gramin", "vikram"],
    "compare": ["compare", "cheaper", "better", "fastest", "best", "vs", "सस्ता", "तुलना"],
    "greeting": ["hi", "hello", "namaste", "नमस्ते", "hey", "morning", "evening"],
    "help": ["help", "मदद", "feature", "command", "guide", "kya kar sakte"],
    "feedback": ["thanks", "thank", "wrong", "helpful", "धन्यवाद", "shukriya", "galat", "correct"],
}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector *a* and matrix *b*.

    Args:
        a: Query vector of shape ``(dim,)``.
        b: Matrix of shape ``(n, dim)``.

    Returns:
        1-D array of similarities of length *n*.
    """
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b_norm, a_norm)


class IntentClassifier:
    """Embedding-based intent classifier for the Delhi Commute Bot.

    The classifier pre-computes embeddings for a set of example sentences
    per intent using ``sentence-transformers/all-MiniLM-L6-v2``.  At inference
    time the input text is embedded and the intent with the highest average
    cosine similarity is returned.

    If ``sentence-transformers`` is not installed, the classifier falls back to
    a simple keyword-matching strategy.

    Attributes:
        SUPPORTED_INTENTS: Class-level tuple of all supported intent labels.
        model_name: Name/path of the sentence-transformer model.
    """

    SUPPORTED_INTENTS: ClassVar[Tuple[str, ...]] = (
        "bus_route",
        "metro_route",
        "auto_fare",
        "shared_auto",
        "compare",
        "greeting",
        "help",
        "feedback",
        "general",
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        intent_examples: Optional[Dict[str, List[str]]] = None,
        confidence_threshold: float = 0.35,
    ) -> None:
        """Initialise the intent classifier.

        Args:
            model_name: HuggingFace model identifier for the sentence
                transformer.  Defaults to ``all-MiniLM-L6-v2``.
            intent_examples: Optional override for the default intent→examples
                mapping.  If ``None`` the built-in examples are used.
            confidence_threshold: Minimum cosine similarity required for a
                confident classification.  Below this value the intent will
                be returned as ``"general"``.
        """
        self.model_name: str = model_name
        self.confidence_threshold: float = confidence_threshold
        self._intent_examples: Dict[str, List[str]] = (
            intent_examples if intent_examples is not None else _DEFAULT_INTENT_EXAMPLES
        )

        # Populated by _build_index()
        self._model: Any = None
        self._intent_labels: List[str] = []
        self._example_embeddings: Optional[np.ndarray] = None
        self._label_indices: List[int] = []  # maps row → intent index

        self._use_embeddings: bool = _HAS_SENTENCE_TRANSFORMERS
        if self._use_embeddings:
            self._build_index()
        else:
            logger.info("Running in keyword-fallback mode.")

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Load the sentence-transformer model and pre-compute embeddings."""
        logger.info("Loading sentence-transformer model '%s' …", self.model_name)
        self._model = SentenceTransformer(self.model_name)

        all_sentences: List[str] = []
        label_indices: List[int] = []
        intent_labels: List[str] = []

        for idx, (intent, examples) in enumerate(self._intent_examples.items()):
            intent_labels.append(intent)
            for sentence in examples:
                all_sentences.append(sentence)
                label_indices.append(idx)

        logger.info(
            "Encoding %d example sentences across %d intents …",
            len(all_sentences),
            len(intent_labels),
        )

        self._example_embeddings = self._model.encode(
            all_sentences,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self._intent_labels = intent_labels
        self._label_indices = label_indices
        logger.info("Intent index built successfully.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Dict[str, Any]:
        """Classify the user's input text into an intent.

        Args:
            text: Raw user input string.

        Returns:
            A dictionary with keys:
                - ``intent`` (*str*): The predicted intent label.
                - ``confidence`` (*float*): Confidence score in ``[0, 1]``.
        """
        if not text or not text.strip():
            return {"intent": "general", "confidence": 0.0}

        if self._use_embeddings and self._example_embeddings is not None:
            return self._predict_embedding(text)
        return self._predict_keyword(text)

    def _predict_embedding(self, text: str) -> Dict[str, Any]:
        """Predict intent using cosine similarity against example embeddings.

        For each intent we compute the *mean* cosine similarity of the query
        against all that intent's examples, then pick the intent with the
        highest mean similarity.
        """
        query_embedding: np.ndarray = self._model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        similarities = _cosine_similarity(query_embedding, self._example_embeddings)  # type: ignore[arg-type]

        # Aggregate: mean similarity per intent
        n_intents = len(self._intent_labels)
        intent_scores = np.zeros(n_intents, dtype=np.float64)
        intent_counts = np.zeros(n_intents, dtype=np.float64)

        for sim, label_idx in zip(similarities, self._label_indices):
            intent_scores[label_idx] += sim
            intent_counts[label_idx] += 1

        # Avoid division by zero
        intent_counts[intent_counts == 0] = 1
        mean_scores = intent_scores / intent_counts

        best_idx = int(np.argmax(mean_scores))
        best_score = float(mean_scores[best_idx])

        # If below threshold, classify as "general"
        if best_score < self.confidence_threshold:
            return {"intent": "general", "confidence": round(best_score, 4)}

        return {
            "intent": self._intent_labels[best_idx],
            "confidence": round(best_score, 4),
        }

    def _predict_keyword(self, text: str) -> Dict[str, Any]:
        """Fallback keyword-based prediction when embeddings are unavailable.

        Counts matching keywords per intent and picks the one with the most
        matches.  Confidence is approximated as ``matches / total_keywords``.
        """
        text_lower = text.lower()
        best_intent = "general"
        best_score = 0.0

        for intent, keywords in _KEYWORD_FALLBACK.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                score = matches / len(keywords)
                if score > best_score:
                    best_score = score
                    best_intent = intent

        return {
            "intent": best_intent,
            "confidence": round(best_score, 4),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the classifier state to disk.

        The method creates/overwrites a directory at *path* containing:
          - ``config.json`` – model name, threshold, intent examples.
          - ``embeddings.pkl`` – pre-computed numpy embeddings and labels.

        Args:
            path: Directory path to save the classifier into.
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "intent_examples": self._intent_examples,
        }
        with open(save_dir / "config.json", "w", encoding="utf-8") as fh:
            json.dump(config, fh, ensure_ascii=False, indent=2)

        embedding_data = {
            "embeddings": self._example_embeddings,
            "intent_labels": self._intent_labels,
            "label_indices": self._label_indices,
        }
        with open(save_dir / "embeddings.pkl", "wb") as fh:
            pickle.dump(embedding_data, fh)

        logger.info("IntentClassifier saved to %s", save_dir)

    @classmethod
    def load(cls, path: str | Path) -> "IntentClassifier":
        """Load a previously saved classifier from disk.

        This avoids re-encoding all example sentences.  The sentence-
        transformer model is still loaded for runtime inference, but the
        pre-computed example embeddings are restored from the pickle file.

        Args:
            path: Directory path that was passed to :meth:`save`.

        Returns:
            A fully initialised ``IntentClassifier`` instance.

        Raises:
            FileNotFoundError: If *path* does not contain the expected files.
        """
        load_dir = Path(path)

        config_path = load_dir / "config.json"
        embeddings_path = load_dir / "embeddings.pkl"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)

        # Create instance *without* auto-building the index
        instance = cls.__new__(cls)
        instance.model_name = config["model_name"]
        instance.confidence_threshold = config.get("confidence_threshold", 0.35)
        instance._intent_examples = config.get("intent_examples", _DEFAULT_INTENT_EXAMPLES)
        instance._use_embeddings = _HAS_SENTENCE_TRANSFORMERS

        if _HAS_SENTENCE_TRANSFORMERS:
            instance._model = SentenceTransformer(instance.model_name)
        else:
            instance._model = None

        # Restore pre-computed embeddings if available
        if embeddings_path.exists():
            with open(embeddings_path, "rb") as fh:
                emb_data = pickle.load(fh)
            instance._example_embeddings = emb_data["embeddings"]
            instance._intent_labels = emb_data["intent_labels"]
            instance._label_indices = emb_data["label_indices"]
        else:
            logger.warning(
                "Embeddings file not found at %s – rebuilding index.", embeddings_path
            )
            instance._example_embeddings = None
            instance._intent_labels = []
            instance._label_indices = []
            if instance._use_embeddings:
                instance._build_index()

        logger.info("IntentClassifier loaded from %s", load_dir)
        return instance

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def add_examples(self, intent: str, examples: List[str]) -> None:
        """Add extra training examples for an intent and rebuild the index.

        Args:
            intent: Intent label (must be in :attr:`SUPPORTED_INTENTS`).
            examples: List of example sentences.

        Raises:
            ValueError: If *intent* is not a supported intent label.
        """
        if intent not in self.SUPPORTED_INTENTS:
            raise ValueError(
                f"Unknown intent '{intent}'. "
                f"Supported: {self.SUPPORTED_INTENTS}"
            )

        self._intent_examples.setdefault(intent, []).extend(examples)

        if self._use_embeddings:
            self._build_index()

    def __repr__(self) -> str:
        mode = "embedding" if self._use_embeddings else "keyword-fallback"
        n_examples = sum(len(v) for v in self._intent_examples.values())
        return (
            f"IntentClassifier(model={self.model_name!r}, "
            f"mode={mode}, "
            f"intents={len(self._intent_examples)}, "
            f"examples={n_examples})"
        )
