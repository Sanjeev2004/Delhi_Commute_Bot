"""
WhatsApp-friendly response formatter (rule-based, no LLM required).

Generates rich text responses with emojis and clean formatting for
each transport mode.  All public methods return plain ``str`` ready
to be sent over the WhatsApp API.

Supports English, Hindi, and Hinglish responses.
"""

from __future__ import annotations

from typing import Any

from src.api.schemas import TransportOption


# ── Shared helpers ──────────────────────────────────────────────────────────

_DIVIDER = "━━━━━━━━━━━━━━━━━━━"


def _crowding_emoji(level: str | None) -> str:
    """Return an emoji for the crowding level."""
    match (level or "").lower():
        case "low":
            return "🟢 Low"
        case "moderate":
            return "🟡 Moderate"
        case "high":
            return "🔴 High"
        case _:
            return "⚪ N/A"


# ── ResponseFormatter ──────────────────────────────────────────────────────


class ResponseFormatter:
    """Rule-based response formatter producing WhatsApp-ready messages."""

    # ── Bus ──────────────────────────────────────────────────────────────

    @staticmethod
    def format_bus_response(
        routes: list[dict[str, Any]],
        source: str,
        dest: str,
    ) -> tuple[str, list[TransportOption]]:
        """Format bus route results.

        Parameters
        ----------
        routes:
            List of dicts with keys like ``route_id``, ``name``, ``via``,
            ``fare_inr``, ``crowding``, ``estimated_time``.
        source, dest:
            Human-readable source / destination names.

        Returns
        -------
        tuple[str, list[TransportOption]]
            WhatsApp message text and structured ``TransportOption`` list.
        """
        if not routes:
            return (
                f"🚌 *Bus: {source} → {dest}*\n\n"
                f"Sorry, I couldn't find specific bus routes for this pair.\n"
                f"कोई बस रूट नहीं मिला इस रूट के लिए।\n\n"
                f"💡 *Tip:* Try checking the DTC app or ask about Metro / Auto options!",
                [],
            )

        lines: list[str] = [f"🚌 *DTC Bus: {source} → {dest}*\n"]
        options: list[TransportOption] = []

        for i, r in enumerate(routes[:3]):
            route_id = r.get("route_id", "?")
            via = r.get("via", "")
            fare = r.get("fare_inr")
            crowding = r.get("crowding")
            est_time = r.get("estimated_time")

            via_str = f" (via {via})" if via else ""
            lines.append(f"📍 *Route {route_id}*{via_str}")
            if est_time:
                lines.append(f"  🕐 Estimated: {est_time}")
            lines.append(f"  👥 Crowding: {_crowding_emoji(crowding)}")
            if fare is not None:
                lines.append(f"  💰 Fare: ₹{fare}")
            lines.append("")

            options.append(
                TransportOption(
                    mode="bus",
                    route_info=f"Route {route_id}{via_str}",
                    estimated_time=est_time,
                    fare_inr=fare,
                    crowding=crowding,
                )
            )

        lines.append("💡 DTC बसों में स्मार्ट कार्ड से 10% छूट मिलती है!")
        return "\n".join(lines), options

    # ── Auto ─────────────────────────────────────────────────────────────

    @staticmethod
    def format_auto_response(
        fare_info: dict[str, Any],
        source: str,
        dest: str,
    ) -> tuple[str, list[TransportOption]]:
        """Format auto-rickshaw fare information.

        Parameters
        ----------
        fare_info:
            Dict with keys ``meter_fare_inr``, ``asking_fare_inr``,
            ``distance_km``, optionally ``night_fare_inr``.
        """
        meter = fare_info.get("meter_fare_inr", "?")
        asking = fare_info.get("asking_fare_inr")
        distance = fare_info.get("distance_km")
        night = fare_info.get("night_fare_inr")

        lines = [f"🛺 *Auto: {source} → {dest}*\n"]
        if distance:
            lines.append(f"📏 Distance: ~{distance} km")
        lines.append(f"💰 Meter fare: ₹{meter}")
        if asking:
            lines.append(f"💸 Typical asking: ₹{asking}")
        if night:
            lines.append(f"🌙 Night fare (11PM-5AM): ₹{night}")
        lines.append("")
        lines.append(
            "💡 *Tip:* Always insist on meter! मीटर पर ही जाएं।\n"
            "📞 Report overcharging: 011-42400400"
        )

        option = TransportOption(
            mode="auto",
            route_info=f"{source} → {dest}",
            estimated_time=fare_info.get("estimated_time"),
            fare_inr=int(meter) if str(meter).isdigit() else None,
            crowding=None,
        )
        return "\n".join(lines), [option]

    # ── Metro ────────────────────────────────────────────────────────────

    @staticmethod
    def format_metro_response(
        path_info: dict[str, Any],
        source: str,
        dest: str,
    ) -> tuple[str, list[TransportOption]]:
        """Format metro route information.

        Parameters
        ----------
        path_info:
            Dict with keys ``line``, ``stations``, ``interchanges``,
            ``fare_inr``, ``estimated_time``, ``walking_time``.
        """
        line = path_info.get("line", "?")
        stations = path_info.get("stations", 0)
        fare = path_info.get("fare_inr")
        est_time = path_info.get("estimated_time")
        walk = path_info.get("walking_time")
        interchanges = path_info.get("interchanges", [])

        lines = [f"🚇 *Metro: {source} → {dest}*\n"]
        lines.append(f"🔵 Line: {line}")
        lines.append(f"🔢 Stations: {stations}")
        if interchanges:
            lines.append(f"🔄 Interchange: {', '.join(interchanges)}")
        if fare is not None:
            lines.append(f"💰 Fare: ₹{fare}")
        if est_time:
            time_str = est_time
            if walk:
                time_str += f" + {walk} walk"
            lines.append(f"⏱ Time: {time_str}")
        lines.append("")
        lines.append(
            "💡 *Tip:* स्मार्ट कार्ड से ~10% छूट!\n"
            "🕐 First train: 6:00 AM | Last train: 11:00 PM"
        )

        option = TransportOption(
            mode="metro",
            route_info=f"{line}, {stations} stops",
            estimated_time=est_time,
            fare_inr=fare,
            crowding=None,
        )
        return "\n".join(lines), [option]

    # ── Compare ──────────────────────────────────────────────────────────

    @staticmethod
    def format_compare_response(
        all_options: list[dict[str, Any]],
        source: str,
        dest: str,
    ) -> tuple[str, list[TransportOption]]:
        """Compare multiple transport modes side by side.

        Parameters
        ----------
        all_options:
            List of dicts, each with keys ``mode``, ``route_info``,
            ``estimated_time``, ``fare_inr``, ``crowding``.
        """
        mode_emoji = {
            "bus": "🚌",
            "metro": "🚇",
            "auto": "🛺",
            "shared_auto": "🛺",
            "e_rickshaw": "🛺",
        }

        lines = [f"🔄 *Compare: {source} → {dest}*\n"]
        structured_options: list[TransportOption] = []

        for opt in all_options:
            mode = opt.get("mode", "unknown")
            emoji = mode_emoji.get(mode, "🚗")
            fare = opt.get("fare_inr")
            est = opt.get("estimated_time", "?")
            info = opt.get("route_info", "")

            fare_str = f"₹{fare}" if fare else "?"
            lines.append(f"{emoji} *{mode.replace('_', ' ').title()}*: {fare_str}")
            if info:
                lines.append(f"   {info}")
            lines.append(f"   ⏱ {est}")
            lines.append("")

            structured_options.append(
                TransportOption(
                    mode=mode,
                    route_info=info,
                    estimated_time=est if est != "?" else None,
                    fare_inr=fare,
                    crowding=opt.get("crowding"),
                )
            )

        # Find cheapest
        fares = [(o.mode, o.fare_inr) for o in structured_options if o.fare_inr]
        if fares:
            cheapest = min(fares, key=lambda x: x[1])
            lines.append(
                f"💡 *Cheapest / सबसे सस्ता:* {cheapest[0].replace('_', ' ').title()} (₹{cheapest[1]})"
            )

        return "\n".join(lines), structured_options

    # ── Shared auto ──────────────────────────────────────────────────────

    @staticmethod
    def format_shared_auto_response(
        routes: list[dict[str, Any]],
    ) -> tuple[str, list[TransportOption]]:
        """Format shared auto / e-rickshaw route information.

        Parameters
        ----------
        routes:
            List of route dicts from the shared_auto dataset.
        """
        if not routes:
            return (
                "🛺 *Shared Auto / E-Rickshaw*\n\n"
                "No shared auto routes found for this area.\n"
                "इस इलाके में शेयर ऑटो नहीं मिला।\n\n"
                "💡 *Tip:* Shared autos are mostly found near metro stations.",
                [],
            )

        lines = ["🛺 *Shared Auto / E-Rickshaw Routes*\n"]
        options: list[TransportOption] = []

        for r in routes[:5]:
            name = r.get("name", "")
            frm = r.get("from", "?")
            to = r.get("to", "?")
            fare = r.get("fare_inr")
            freq = r.get("frequency", "")
            hours = r.get("operating_hours", "")
            rtype = r.get("type", "shared_auto")

            type_emoji = "🛺" if rtype == "shared_auto" else "🔋"
            lines.append(f"{type_emoji} *{name}*")
            lines.append(f"   📍 {frm} → {to}")
            if fare is not None:
                lines.append(f"   💰 Fare: ₹{fare}")
            if freq:
                lines.append(f"   ⏱ Frequency: {freq}")
            if hours:
                lines.append(f"   🕐 Hours: {hours}")
            lines.append("")

            options.append(
                TransportOption(
                    mode=rtype,
                    route_info=f"{frm} → {to}",
                    estimated_time=None,
                    fare_inr=fare,
                    crowding=None,
                )
            )

        return "\n".join(lines), options

    # ── Greeting ─────────────────────────────────────────────────────────

    @staticmethod
    def format_greeting_response() -> str:
        """Return a friendly greeting with usage instructions."""
        return (
            "🙏 *नमस्ते! Welcome to DelhiCommuteBot!*\n\n"
            "I can help you with Delhi transport info:\n\n"
            "🚌 *Bus Routes* — \"Bus from CP to AIIMS\"\n"
            "🚇 *Metro Routes* — \"Metro to Rajiv Chowk\"\n"
            "🛺 *Auto Fares* — \"Auto fare from Saket to Hauz Khas\"\n"
            "🔄 *Compare* — \"Compare options Dwarka to CP\"\n"
            "🛺 *Shared Auto* — \"Shared auto near Laxmi Nagar\"\n\n"
            "💬 बस अपना सवाल पूछें — Hindi, Hinglish, or English!\n\n"
            "Type *help* for more details."
        )

    # ── Help ─────────────────────────────────────────────────────────────

    @staticmethod
    def format_help_response() -> str:
        """Return detailed help text with examples."""
        return (
            "ℹ️ *DelhiCommuteBot — Help / मदद*\n\n"
            f"{_DIVIDER}\n"
            "🚌 *Bus Routes*\n"
            "  • \"Bus from Kashmere Gate to Laxmi Nagar\"\n"
            "  • \"कश्मीरी गेट से बस कौन सी जाएगी\"\n"
            "  • \"DTC bus number for Dwarka\"\n\n"
            "🚇 *Metro Routes*\n"
            "  • \"Metro route from Dwarka to Rajiv Chowk\"\n"
            "  • \"मेट्रो कैसे जाऊं हौज़ खास\"\n"
            "  • \"DMRC route to CP\"\n\n"
            "🛺 *Auto Fares*\n"
            "  • \"Auto fare from CP to Saket\"\n"
            "  • \"ऑटो का किराया बताओ\"\n"
            "  • \"Rickshaw cost Nehru Place to AIIMS\"\n\n"
            "🔄 *Compare Options*\n"
            "  • \"Compare options from Nehru Place to AIIMS\"\n"
            "  • \"Which is cheaper, bus or metro?\"\n\n"
            "🛺 *Shared Auto / E-Rickshaw*\n"
            "  • \"Shared auto near Uttam Nagar metro\"\n"
            "  • \"E-rickshaw routes from Laxmi Nagar\"\n\n"
            f"{_DIVIDER}\n"
            "💡 *Tips:*\n"
            "  • Always mention *from* and *to* locations\n"
            "  • You can ask follow-up questions!\n"
            "  • Works in English, Hindi & Hinglish\n"
            "  • Type *thanks* or *👍* to give feedback"
        )

    # ── Feedback Acknowledgement ─────────────────────────────────────────

    @staticmethod
    def format_feedback_ack_response() -> str:
        """Acknowledge user feedback."""
        return (
            "🙏 *Thank you for your feedback!*\n"
            "धन्यवाद! आपकी राय हमारे लिए महत्वपूर्ण है।\n\n"
            "Feel free to ask me another commute query anytime! 🚌🚇🛺"
        )

    # ── Fallback ─────────────────────────────────────────────────────────

    @staticmethod
    def format_fallback_response(raw_query: str) -> str:
        """Return a friendly fallback when intent is unclear."""
        return (
            "🤔 I'm not sure I understood that.\n"
            "मुझे समझ नहीं आया।\n\n"
            "Try asking me something like:\n"
            '• "Bus from Kashmere Gate to Laxmi Nagar"\n'
            '• "Auto fare from CP to Saket"\n'
            '• "Metro route Dwarka to Rajiv Chowk"\n'
            '• "Compare options from Nehru Place to AIIMS"\n'
            '• "Shared auto near Uttam Nagar"\n\n'
            "💡 I work best with a *from* and *to* location!\n"
            "कृपया *from* और *to* location बताएं!"
        )
