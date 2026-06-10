"""
Twilio WhatsApp integration service for DelhiCommuteBot.

Provides:
- Webhook signature verification (manual HMAC fallback when ``twilio``
  package is not installed).
- Async message sending via ``httpx``.
- WhatsApp text formatting with character-limit enforcement.

The service reads credentials from :pydata:`src.config.settings` and
is designed to degrade gracefully when Twilio credentials are missing
(e.g. in local development).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import math
from urllib.parse import urlencode
from typing import Any

import httpx
from loguru import logger

from src.config import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHATSAPP_CHAR_LIMIT: int = 1600
"""Maximum characters per single WhatsApp message."""

TWILIO_API_BASE: str = "https://api.twilio.com/2010-04-01"
"""Base URL for the Twilio REST API."""

# ---------------------------------------------------------------------------
# Attempt to import the official Twilio helper for signature validation
# ---------------------------------------------------------------------------

_twilio_available: bool = False

try:
    from twilio.request_validator import RequestValidator as _TwilioValidator  # type: ignore[import-untyped]

    _twilio_available = True
    logger.debug("Official Twilio SDK detected — using native RequestValidator")
except ImportError:
    logger.info(
        "Twilio SDK not installed — falling back to manual HMAC webhook verification"
    )


# ---------------------------------------------------------------------------
# Manual HMAC signature verification (fallback)
# ---------------------------------------------------------------------------

def _manual_compute_signature(auth_token: str, url: str, params: dict[str, str]) -> str:
    """Compute a Twilio-compatible request signature using HMAC-SHA1.

    The algorithm matches Twilio's specification:
    1. Take the full URL of the request.
    2. Append query/form parameters sorted alphabetically by key,
       concatenating key-value pairs without separators.
    3. Compute HMAC-SHA1 of the resulting string using the auth token
       as the key.
    4. Base64-encode the digest.

    Parameters
    ----------
    auth_token:
        Twilio account auth token used as the HMAC key.
    url:
        The full request URL (including scheme and host).
    params:
        Form-encoded body parameters as a flat dictionary.

    Returns
    -------
    str
        The Base64-encoded signature string.
    """
    data_to_sign = url
    if params:
        for key in sorted(params.keys()):
            data_to_sign += key + params[key]

    digest = hmac.new(
        auth_token.encode("utf-8"),
        data_to_sign.encode("utf-8"),
        hashlib.sha1,
    ).digest()

    return base64.b64encode(digest).decode("utf-8")


# ---------------------------------------------------------------------------
# TwilioService
# ---------------------------------------------------------------------------

class TwilioService:
    """High-level service for Twilio WhatsApp interactions.

    Usage::

        svc = TwilioService()

        # Verify an incoming webhook
        is_valid = svc.verify_webhook_signature(
            request_url="https://example.com/webhook",
            params={"Body": "hello", "From": "whatsapp:+91..."},
            signature="abc123==",
        )

        # Send a reply
        await svc.send_whatsapp_message(
            to="whatsapp:+919876543210",
            body="Hello from DelhiCommuteBot!",
        )
    """

    def __init__(
        self,
        account_sid: str | None = None,
        auth_token: str | None = None,
        whatsapp_number: str | None = None,
    ) -> None:
        """Initialise with explicit creds or fall back to settings.

        Parameters
        ----------
        account_sid:
            Twilio Account SID.  Falls back to ``settings.twilio_account_sid``.
        auth_token:
            Twilio Auth Token.  Falls back to ``settings.twilio_auth_token``.
        whatsapp_number:
            The Twilio WhatsApp sender number (e.g. ``whatsapp:+14155238886``).
            Falls back to ``settings.twilio_whatsapp_number``.
        """
        self.account_sid: str = account_sid or settings.twilio_account_sid
        self.auth_token: str = auth_token or settings.twilio_auth_token
        self.whatsapp_number: str = whatsapp_number or settings.twilio_whatsapp_number

        # Native Twilio validator (if the SDK is installed)
        self._native_validator: _TwilioValidator | None = None  # type: ignore[name-defined]
        if _twilio_available and self.auth_token:
            self._native_validator = _TwilioValidator(self.auth_token)

        if not self.account_sid or not self.auth_token:
            logger.warning(
                "Twilio credentials not configured — "
                "webhook verification and messaging will be disabled"
            )

    # ------------------------------------------------------------------
    # Webhook verification
    # ------------------------------------------------------------------

    def verify_webhook_signature(
        self,
        request_url: str,
        params: dict[str, str],
        signature: str,
        auth_token: str | None = None,
    ) -> bool:
        """Verify that an incoming request was signed by Twilio.

        Uses the official ``twilio`` SDK if available; otherwise falls
        back to a manual HMAC-SHA1 computation that matches Twilio's
        signing algorithm.

        Parameters
        ----------
        request_url:
            The full URL that Twilio sent the request to.
        params:
            The POST body parameters as a flat ``dict``.
        signature:
            Value of the ``X-Twilio-Signature`` header.
        auth_token:
            Override auth token (defaults to instance token).

        Returns
        -------
        bool
            ``True`` if the signature is valid.
        """
        token = auth_token or self.auth_token

        if not token:
            logger.warning(
                "No auth token available — skipping webhook verification "
                "(accepting request in dev mode)"
            )
            return True

        if self._native_validator is not None:
            is_valid: bool = self._native_validator.validate(
                request_url, params, signature,
            )
            logger.debug(
                "Twilio SDK webhook verification result: {}", is_valid,
            )
            return is_valid

        # Fallback: manual HMAC verification
        computed = _manual_compute_signature(token, request_url, params)
        is_valid = hmac.compare_digest(computed, signature)
        logger.debug(
            "Manual HMAC webhook verification result: {} "
            "(computed={}, received={})",
            is_valid,
            computed[:12] + "…",
            signature[:12] + "…",
        )
        return is_valid

    # ------------------------------------------------------------------
    # Message sending (async)
    # ------------------------------------------------------------------

    async def send_whatsapp_message(
        self,
        to: str,
        body: str,
        *,
        media_url: str | None = None,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Send a WhatsApp message via the Twilio REST API.

        If the message exceeds :pydata:`WHATSAPP_CHAR_LIMIT`, it is
        automatically split into multiple messages.

        Parameters
        ----------
        to:
            Destination in ``whatsapp:+<E.164>`` format.
        body:
            The message text.
        media_url:
            Optional URL of a media attachment.
        timeout:
            HTTP request timeout in seconds.

        Returns
        -------
        dict
            Twilio API response JSON for the last sent message, or an
            error dict on failure.

        Raises
        ------
        RuntimeError
            If Twilio credentials are not configured.
        """
        if not self.account_sid or not self.auth_token:
            error_msg = (
                "Cannot send WhatsApp message — "
                "Twilio credentials not configured"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Normalise the destination number
        if not to.startswith("whatsapp:"):
            to = f"whatsapp:{to}"

        # Split long messages
        chunks = self._split_message(body)
        last_response: dict[str, Any] = {}

        url = (
            f"{TWILIO_API_BASE}/Accounts/{self.account_sid}/Messages.json"
        )
        auth = (self.account_sid, self.auth_token)

        async with httpx.AsyncClient(timeout=timeout) as client:
            for idx, chunk in enumerate(chunks):
                payload: dict[str, str] = {
                    "From": self.whatsapp_number,
                    "To": to,
                    "Body": chunk,
                }
                if media_url and idx == 0:
                    payload["MediaUrl"] = media_url

                try:
                    response = await client.post(
                        url,
                        data=payload,
                        auth=auth,
                    )
                    response.raise_for_status()
                    last_response = response.json()
                    logger.info(
                        "WhatsApp message sent to {} (sid={}, part {}/{})",
                        to,
                        last_response.get("sid", "unknown"),
                        idx + 1,
                        len(chunks),
                    )
                except httpx.HTTPStatusError as exc:
                    error_body = exc.response.text
                    logger.error(
                        "Twilio API error (HTTP {}): {}",
                        exc.response.status_code,
                        error_body,
                    )
                    last_response = {
                        "error": True,
                        "status_code": exc.response.status_code,
                        "detail": error_body,
                    }
                except httpx.RequestError as exc:
                    logger.error("HTTP request to Twilio failed: {}", exc)
                    last_response = {
                        "error": True,
                        "detail": str(exc),
                    }

        return last_response

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_for_whatsapp(text: str) -> str:
        """Sanitise and truncate *text* to fit within WhatsApp limits.

        Rules applied:
        1. Strip leading/trailing whitespace.
        2. Replace consecutive blank lines with a single blank line.
        3. Truncate to :pydata:`WHATSAPP_CHAR_LIMIT` characters, appending
           an ellipsis indicator if truncated.

        Parameters
        ----------
        text:
            Raw text to format.

        Returns
        -------
        str
            Cleaned text that fits within a single WhatsApp message.
        """
        if not text:
            return ""

        # Clean up whitespace
        text = text.strip()

        # Collapse multiple blank lines
        lines = text.splitlines()
        cleaned: list[str] = []
        prev_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            cleaned.append(line)
            prev_blank = is_blank
        text = "\n".join(cleaned)

        # Truncate if necessary
        if len(text) > WHATSAPP_CHAR_LIMIT:
            truncation_marker = "\n\n… (message truncated)"
            max_len = WHATSAPP_CHAR_LIMIT - len(truncation_marker)
            # Try to break at a newline or space
            cut_point = text.rfind("\n", 0, max_len)
            if cut_point < max_len // 2:
                cut_point = text.rfind(" ", 0, max_len)
            if cut_point < max_len // 2:
                cut_point = max_len
            text = text[:cut_point] + truncation_marker

        return text

    @staticmethod
    def _split_message(text: str) -> list[str]:
        """Split *text* into chunks that each fit within the WhatsApp limit.

        Tries to break on paragraph boundaries (double newlines), then
        single newlines, then spaces.

        Parameters
        ----------
        text:
            The full message text.

        Returns
        -------
        list[str]
            Ordered list of message chunks.
        """
        text = TwilioService.format_for_whatsapp(text)
        if len(text) <= WHATSAPP_CHAR_LIMIT:
            return [text]

        chunks: list[str] = []
        remaining = text

        while remaining:
            if len(remaining) <= WHATSAPP_CHAR_LIMIT:
                chunks.append(remaining)
                break

            # Find a good split point
            split_at = remaining.rfind("\n\n", 0, WHATSAPP_CHAR_LIMIT)
            if split_at < WHATSAPP_CHAR_LIMIT // 3:
                split_at = remaining.rfind("\n", 0, WHATSAPP_CHAR_LIMIT)
            if split_at < WHATSAPP_CHAR_LIMIT // 3:
                split_at = remaining.rfind(" ", 0, WHATSAPP_CHAR_LIMIT)
            if split_at < WHATSAPP_CHAR_LIMIT // 3:
                split_at = WHATSAPP_CHAR_LIMIT

            chunks.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()

        # Add part indicators for multi-part messages
        if len(chunks) > 1:
            total = len(chunks)
            chunks = [
                f"[{i + 1}/{total}] {chunk}"
                for i, chunk in enumerate(chunks)
            ]

        return chunks
