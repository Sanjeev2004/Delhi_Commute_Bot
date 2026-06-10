"""
Tests for the response formatter (src.rag.chains.ResponseFormatter).

These tests verify that WhatsApp-ready messages are formatted correctly
for every transport mode, including edge cases like empty data and
emoji presence.
"""

from __future__ import annotations

import pytest

from src.rag.chains import ResponseFormatter


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def formatter() -> ResponseFormatter:
    """Return a fresh ResponseFormatter instance."""
    return ResponseFormatter()


# ---------------------------------------------------------------------------
# Bus response tests
# ---------------------------------------------------------------------------


class TestBusResponse:
    """Tests for ``format_bus_response``."""

    def test_bus_response_with_routes(self, formatter: ResponseFormatter) -> None:
        """With valid routes the response should contain route IDs."""
        routes = [
            {
                "route_id": "534",
                "name": "Kashmere Gate - Laxmi Nagar",
                "via": "ISBT",
                "fare_inr": 15,
                "crowding": "moderate",
                "estimated_time": "30 min",
            },
            {
                "route_id": "473",
                "name": "Kashmere Gate - Laxmi Nagar",
                "via": "Preet Vihar",
                "fare_inr": 10,
                "crowding": "low",
                "estimated_time": "25 min",
            },
        ]
        text, options = formatter.format_bus_response(
            routes, "Kashmere Gate", "Laxmi Nagar"
        )
        assert "534" in text, "Response should contain route ID '534'"
        assert "473" in text, "Response should contain route ID '473'"
        assert len(options) == 2, f"Expected 2 options, got {len(options)}"
        assert options[0].mode == "bus", f"Expected mode='bus', got '{options[0].mode}'"

    def test_bus_response_empty(self, formatter: ResponseFormatter) -> None:
        """With no routes the response should contain a fallback message."""
        text, options = formatter.format_bus_response([], "CP", "AIIMS")
        assert len(options) == 0, f"Expected 0 options, got {len(options)}"
        assert "sorry" in text.lower() or "couldn't find" in text.lower(), (
            "Empty bus response should contain a fallback / apology message"
        )


# ---------------------------------------------------------------------------
# Metro response tests
# ---------------------------------------------------------------------------


class TestMetroResponse:
    """Tests for ``format_metro_response``."""

    def test_metro_response(self, formatter: ResponseFormatter) -> None:
        """Response should contain line name, stations, and fare."""
        path_info = {
            "line": "Blue Line",
            "stations": 8,
            "interchanges": ["Rajiv Chowk"],
            "fare_inr": 30,
            "estimated_time": "20 min",
            "walking_time": "5 min",
        }
        text, options = formatter.format_metro_response(
            path_info, "Dwarka", "Rajiv Chowk"
        )
        assert "Blue Line" in text, "Response should mention the metro line"
        assert "8" in text, "Response should mention station count"
        assert "30" in text, "Response should mention the fare"
        assert len(options) == 1, f"Expected 1 option, got {len(options)}"
        assert options[0].mode == "metro", f"Expected mode='metro', got '{options[0].mode}'"


# ---------------------------------------------------------------------------
# Auto response tests
# ---------------------------------------------------------------------------


class TestAutoResponse:
    """Tests for ``format_auto_response``."""

    def test_auto_response(self, formatter: ResponseFormatter) -> None:
        """Response should contain meter fare and asking fare."""
        fare_info = {
            "meter_fare_inr": 85,
            "asking_fare_inr": 120,
            "distance_km": 7,
            "estimated_time": "25 min",
        }
        text, options = formatter.format_auto_response(fare_info, "CP", "Saket")
        assert "85" in text, "Response should contain meter fare (85)"
        assert "120" in text, "Response should contain asking fare (120)"
        assert len(options) == 1, f"Expected 1 option, got {len(options)}"
        assert options[0].mode == "auto", f"Expected mode='auto', got '{options[0].mode}'"


# ---------------------------------------------------------------------------
# Compare response tests
# ---------------------------------------------------------------------------


class TestCompareResponse:
    """Tests for ``format_compare_response``."""

    def test_compare_response(self, formatter: ResponseFormatter) -> None:
        """Comparison should mention cheapest mode."""
        all_options = [
            {
                "mode": "bus",
                "route_info": "Route 534",
                "fare_inr": 15,
                "estimated_time": "30 min",
            },
            {
                "mode": "metro",
                "route_info": "Blue Line, 8 stops",
                "fare_inr": 30,
                "estimated_time": "20 min",
            },
            {
                "mode": "auto",
                "route_info": "CP → Saket",
                "fare_inr": 85,
                "estimated_time": "25 min",
            },
        ]
        text, options = formatter.format_compare_response(
            all_options, "CP", "Saket"
        )
        assert "cheapest" in text.lower(), (
            "Compare response should mention the cheapest option"
        )
        assert len(options) == 3, f"Expected 3 options, got {len(options)}"


# ---------------------------------------------------------------------------
# Shared auto response tests
# ---------------------------------------------------------------------------


class TestSharedAutoResponse:
    """Tests for ``format_shared_auto_response``."""

    def test_shared_auto_response(self, formatter: ResponseFormatter) -> None:
        """Response should contain route info."""
        routes = [
            {
                "name": "Uttam Nagar - Dwarka Mor",
                "from": "Uttam Nagar",
                "to": "Dwarka Mor",
                "fare_inr": 10,
                "type": "shared_auto",
                "frequency": "Every 5 min",
                "operating_hours": "6AM - 10PM",
            },
        ]
        text, options = formatter.format_shared_auto_response(routes)
        assert "Uttam Nagar" in text, "Response should contain source location"
        assert "Dwarka Mor" in text, "Response should contain destination location"
        assert "10" in text, "Response should mention the fare"
        assert len(options) == 1, f"Expected 1 option, got {len(options)}"


# ---------------------------------------------------------------------------
# Fallback response tests
# ---------------------------------------------------------------------------


class TestFallbackResponse:
    """Tests for ``format_fallback_response``."""

    def test_fallback_response(self, formatter: ResponseFormatter) -> None:
        """Fallback should contain help text."""
        text = formatter.format_fallback_response("gibberish input")
        assert "try asking" in text.lower() or "not sure" in text.lower(), (
            "Fallback response should provide help text"
        )
        assert "bus" in text.lower(), "Fallback should mention 'bus' as an example"
        assert "metro" in text.lower(), "Fallback should mention 'metro' as an example"


# ---------------------------------------------------------------------------
# Emoji tests
# ---------------------------------------------------------------------------


class TestWhatsAppEmoji:
    """All WhatsApp responses should contain relevant emojis."""

    @pytest.mark.parametrize(
        "method,args,expected_emoji",
        [
            (
                "format_bus_response",
                (
                    [{"route_id": "1", "name": "Test", "via": "", "fare_inr": 10, "crowding": "low", "estimated_time": "10 min"}],
                    "A", "B",
                ),
                "🚌",
            ),
            (
                "format_metro_response",
                (
                    {"line": "Red", "stations": 3, "interchanges": [], "fare_inr": 20, "estimated_time": "15 min", "walking_time": None},
                    "A", "B",
                ),
                "🚇",
            ),
            (
                "format_auto_response",
                (
                    {"meter_fare_inr": 50, "asking_fare_inr": 70, "distance_km": 4, "estimated_time": None},
                    "A", "B",
                ),
                "🛺",
            ),
        ],
        ids=["bus_emoji", "metro_emoji", "auto_emoji"],
    )
    def test_whatsapp_emoji_present(
        self,
        formatter: ResponseFormatter,
        method: str,
        args: tuple,
        expected_emoji: str,
    ) -> None:
        """Each transport mode's response should contain its representative emoji."""
        func = getattr(formatter, method)
        text, _ = func(*args)
        assert expected_emoji in text, (
            f"Expected emoji '{expected_emoji}' in {method} output"
        )
