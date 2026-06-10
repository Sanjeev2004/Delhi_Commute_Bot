"""
WhatsApp-friendly response formatter (rule-based, no LLM required).

Generates rich text responses with emojis and clean formatting for
each transport mode.  All public methods return plain ``str`` ready
to be sent over the WhatsApp API.
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
            lines.append(f"🌙 Night fare: ₹{night}")
        lines.append("")
        lines.append(
            "💡 *Tip:* Always insist on meter. "
            "Report overcharging to 011-42400400."
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
        lines.append("💡 *Tip:* Use a smart card for ~10% discount!")

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
                f"💡 *Cheapest:* {cheapest[0].replace('_', ' ').title()} (₹{cheapest[1]})"
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

    # ── Fallback ─────────────────────────────────────────────────────────

    @staticmethod
    def format_fallback_response(raw_query: str) -> str:
        """Return a friendly fallback when intent is unclear."""
        return (
            "🤔 I'm not sure I understood that.\n\n"
            "Try asking me something like:\n"
            '• "Bus from Kashmere Gate to Laxmi Nagar"\n'
            '• "Auto fare from CP to Saket"\n'
            '• "Metro route Dwarka to Rajiv Chowk"\n'
            '• "Compare options from Nehru Place to AIIMS"\n'
            '• "Shared auto near Uttam Nagar"\n\n'
            "💡 I work best with a *from* and *to* location!"
        )
