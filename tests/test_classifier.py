"""
Tests for the intent classifier (src.classifier.intent.IntentClassifier).

These tests cover:
- Intent classification for all supported transport modes
- Hindi and Hinglish language support
- Confidence thresholds
- Model save / load round-tripping

When ``sentence-transformers`` is not installed the entire module is
skipped gracefully.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Guard: skip entire module if the classifier cannot be imported
try:
    from src.classifier.intent import IntentClassifier

    _CLASSIFIER_AVAILABLE = True
except ImportError:
    _CLASSIFIER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CLASSIFIER_AVAILABLE,
    reason="IntentClassifier or sentence-transformers not available",
)


# ---------------------------------------------------------------------------
# Shared fixture — a fresh classifier instance
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def classifier() -> IntentClassifier:  # type: ignore[name-defined]
    """Return a default IntentClassifier instance."""
    return IntentClassifier()


# ---------------------------------------------------------------------------
# Basic intent classification tests
# ---------------------------------------------------------------------------


class TestIntentClassification:
    """Validate that free-text queries are classified to the correct intent."""

    def test_classify_bus_query(self, classifier: IntentClassifier) -> None:
        result = classifier.predict("bus from Kashmere Gate to Laxmi Nagar")
        assert result["intent"] == "bus_route", (
            f"Expected 'bus_route', got '{result['intent']}'"
        )

    def test_classify_metro_query(self, classifier: IntentClassifier) -> None:
        result = classifier.predict("metro route to Rajiv Chowk")
        assert result["intent"] == "metro_route", (
            f"Expected 'metro_route', got '{result['intent']}'"
        )

    def test_classify_auto_query(self, classifier: IntentClassifier) -> None:
        result = classifier.predict("auto fare from CP to Saket")
        assert result["intent"] == "auto_fare", (
            f"Expected 'auto_fare', got '{result['intent']}'"
        )

    def test_classify_shared_auto_query(self, classifier: IntentClassifier) -> None:
        result = classifier.predict("shared auto near Uttam Nagar")
        assert result["intent"] == "shared_auto", (
            f"Expected 'shared_auto', got '{result['intent']}'"
        )

    def test_classify_compare_query(self, classifier: IntentClassifier) -> None:
        result = classifier.predict("compare options from Nehru Place to AIIMS")
        assert result["intent"] == "compare", (
            f"Expected 'compare', got '{result['intent']}'"
        )

    def test_classify_greeting(self, classifier: IntentClassifier) -> None:
        result = classifier.predict("hello")
        assert result["intent"] == "greeting", (
            f"Expected 'greeting', got '{result['intent']}'"
        )


# ---------------------------------------------------------------------------
# Multilingual tests
# ---------------------------------------------------------------------------


class TestMultilingualClassification:
    """Hindi and Hinglish queries should still be classified correctly."""

    def test_classify_hindi_query(self, classifier: IntentClassifier) -> None:
        result = classifier.predict("कश्मीरी गेट से बस कौन सी जाएगी")
        assert result["intent"] == "bus_route", (
            f"Expected 'bus_route' for Hindi query, got '{result['intent']}'"
        )

    def test_classify_hinglish_query(self, classifier: IntentClassifier) -> None:
        result = classifier.predict("metro ka rasta batao rajiv chowk")
        assert result["intent"] == "metro_route", (
            f"Expected 'metro_route' for Hinglish query, got '{result['intent']}'"
        )


# ---------------------------------------------------------------------------
# Confidence threshold tests
# ---------------------------------------------------------------------------


class TestConfidenceThreshold:
    """Confidence scores should be reasonable for known intents."""

    @pytest.mark.parametrize(
        "query",
        [
            "bus from Kashmere Gate to Laxmi Nagar",
            "metro route to Rajiv Chowk",
            "auto fare from CP to Saket",
            "compare options from Nehru Place to AIIMS",
        ],
    )
    def test_confidence_above_threshold(
        self, classifier: IntentClassifier, query: str
    ) -> None:
        result = classifier.predict(query)
        assert result["confidence"] >= 0.5, (
            f"Confidence {result['confidence']:.2f} is below 0.5 "
            f"for query: '{query}'"
        )


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------


class TestSaveAndLoad:
    """Verify that a classifier can be saved and reloaded with identical behaviour."""

    def test_save_and_load(self, classifier: IntentClassifier) -> None:
        test_queries = [
            "bus from CP to AIIMS",
            "metro to Rajiv Chowk",
            "auto fare Saket",
        ]

        # Predictions before save
        original_predictions = [classifier.predict(q) for q in test_queries]

        # Save to temp dir, reload, predict again
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_classifier"
            classifier.save(str(save_path))

            loaded = IntentClassifier.load(str(save_path))
            loaded_predictions = [loaded.predict(q) for q in test_queries]

        # Compare
        for orig, loaded_pred, query in zip(
            original_predictions, loaded_predictions, test_queries
        ):
            assert orig["intent"] == loaded_pred["intent"], (
                f"Intent mismatch after load for '{query}': "
                f"{orig['intent']} != {loaded_pred['intent']}"
            )
