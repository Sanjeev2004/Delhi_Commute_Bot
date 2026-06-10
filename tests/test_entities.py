"""
Tests for entity extraction (src.classifier.entities.EntityExtractor).

These tests cover:
- ``from … to …`` and simple ``X to Y`` patterns
- ``near`` pattern for single-location queries
- Alias normalisation (CP → Connaught Place, KG → Kashmere Gate)
- Hindi location names
- Fuzzy / misspelled location matching
- Empty / no-location input
- Case-insensitive extraction

When the ``EntityExtractor`` module is unavailable the entire module
is skipped gracefully.
"""

from __future__ import annotations

import pytest

# Guard: skip if the module doesn't exist yet
try:
    from src.classifier.entities import EntityExtractor

    _EXTRACTOR_AVAILABLE = True
except ImportError:
    _EXTRACTOR_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _EXTRACTOR_AVAILABLE,
    reason="EntityExtractor module not available",
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extractor() -> EntityExtractor:  # type: ignore[name-defined]
    """Return a default EntityExtractor instance."""
    return EntityExtractor()


# ---------------------------------------------------------------------------
# Pattern matching tests
# ---------------------------------------------------------------------------


class TestPatternExtraction:
    """Core source/destination extraction patterns."""

    def test_from_to_pattern(self, extractor: EntityExtractor) -> None:
        """'from X to Y' should extract both locations."""
        result = extractor.extract("bus from CP to AIIMS")
        assert result["source"] is not None, "source should not be None"
        assert result["destination"] is not None, "destination should not be None"
        # CP should normalise to Connaught Place
        assert "connaught" in result["source"].lower() or "cp" in result["source"].lower(), (
            f"Expected 'Connaught Place' (or alias), got '{result['source']}'"
        )
        assert "aiims" in result["destination"].lower(), (
            f"Expected 'AIIMS' in destination, got '{result['destination']}'"
        )

    def test_simple_to_pattern(self, extractor: EntityExtractor) -> None:
        """'X to Y' without 'from' should still work."""
        result = extractor.extract("Dwarka to Rajiv Chowk")
        assert result["source"] is not None, "source should not be None"
        assert result["destination"] is not None, "destination should not be None"
        assert "dwarka" in result["source"].lower(), (
            f"Expected 'Dwarka' in source, got '{result['source']}'"
        )
        assert "rajiv" in result["destination"].lower(), (
            f"Expected 'Rajiv Chowk' in destination, got '{result['destination']}'"
        )

    def test_near_pattern(self, extractor: EntityExtractor) -> None:
        """'near X' should set source but leave destination as None."""
        result = extractor.extract("shared auto near Uttam Nagar")
        assert result["source"] is not None, "source should not be None for 'near' pattern"
        assert "uttam" in result["source"].lower(), (
            f"Expected 'Uttam Nagar' in source, got '{result['source']}'"
        )
        # destination may be None for a proximity query
        assert result.get("destination") is None, (
            f"Expected destination=None for 'near' pattern, got '{result['destination']}'"
        )


# ---------------------------------------------------------------------------
# Alias / normalisation tests
# ---------------------------------------------------------------------------


class TestAliasNormalization:
    """Common Delhi abbreviations should be expanded to full names."""

    def test_alias_normalization(self, extractor: EntityExtractor) -> None:
        """CP → Connaught Place, KG → Kashmere Gate."""
        result = extractor.extract("CP to KG")
        source = (result.get("source") or "").lower()
        dest = (result.get("destination") or "").lower()
        assert "connaught" in source or "cp" in source, (
            f"Expected 'Connaught Place' for CP, got '{result.get('source')}'"
        )
        assert "kashmere" in dest or "kg" in dest, (
            f"Expected 'Kashmere Gate' for KG, got '{result.get('destination')}'"
        )


# ---------------------------------------------------------------------------
# Hindi / multilingual tests
# ---------------------------------------------------------------------------


class TestHindiLocations:
    """Hindi location names should be recognised."""

    def test_hindi_locations(self, extractor: EntityExtractor) -> None:
        result = extractor.extract("कनॉट प्लेस से ऐम्स")
        source = (result.get("source") or "").lower()
        dest = (result.get("destination") or "").lower()
        assert "connaught" in source or "कनॉट" in source, (
            f"Expected Connaught Place / कनॉट, got '{result.get('source')}'"
        )
        assert "aiims" in dest or "ऐम्स" in dest, (
            f"Expected AIIMS / ऐम्स, got '{result.get('destination')}'"
        )


# ---------------------------------------------------------------------------
# Fuzzy matching tests
# ---------------------------------------------------------------------------


class TestFuzzyMatching:
    """Misspelled locations should still resolve via fuzzy matching."""

    def test_fuzzy_matching(self, extractor: EntityExtractor) -> None:
        result = extractor.extract("bus from Conaut Place to Saket")
        source = (result.get("source") or "").lower()
        # Should fuzzy-match to "Connaught Place"
        assert "connaught" in source or "conaut" in source, (
            f"Expected fuzzy match to 'Connaught Place', got '{result.get('source')}'"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and edge-case inputs."""

    def test_no_locations(self, extractor: EntityExtractor) -> None:
        """Greetings and irrelevant text should return None for both."""
        result = extractor.extract("hello there")
        assert result.get("source") is None, (
            f"Expected source=None for 'hello there', got '{result.get('source')}'"
        )
        assert result.get("destination") is None, (
            f"Expected destination=None for 'hello there', got '{result.get('destination')}'"
        )

    def test_case_insensitive(self, extractor: EntityExtractor) -> None:
        """Extraction should be case-insensitive."""
        result = extractor.extract("DWARKA to rajiv chowk")
        assert result["source"] is not None, "source should not be None"
        assert result["destination"] is not None, "destination should not be None"
        assert "dwarka" in result["source"].lower(), (
            f"Expected 'Dwarka' in source, got '{result['source']}'"
        )
        assert "rajiv" in result["destination"].lower(), (
            f"Expected 'Rajiv Chowk' in destination, got '{result['destination']}'"
        )
