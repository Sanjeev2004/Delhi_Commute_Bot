"""
Entity extractor for DelhiCommuteBot.

Extracts source and destination locations from user queries using regex
pattern matching and fuzzy string matching (via ``rapidfuzz``) against a
curated dictionary of Delhi locations.

Usage:
    from src.classifier.entities import EntityExtractor

    extractor = EntityExtractor()
    result = extractor.extract("How do I go from CP to AIIMS by metro?")
    # {"source": "Connaught Place", "destination": "AIIMS"}
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful fallback when rapidfuzz is unavailable
# ---------------------------------------------------------------------------
_HAS_RAPIDFUZZ = False
try:
    from rapidfuzz import fuzz, process  # type: ignore[import-untyped]

    _HAS_RAPIDFUZZ = True
except ImportError:
    logger.warning(
        "rapidfuzz is not installed. "
        "EntityExtractor will use exact matching only."
    )

# ---------------------------------------------------------------------------
# Common abbreviation → canonical name mapping
# ---------------------------------------------------------------------------
_ABBREVIATION_MAP: Dict[str, str] = {
    "cp": "Connaught Place",
    "kg": "Kashmere Gate",
    "ndls": "New Delhi Railway Station",
    "dli": "Old Delhi Railway Station",
    "nzm": "Hazrat Nizamuddin",
    "t1": "IGI Airport Terminal 1",
    "t3": "IGI Airport Terminal 3",
    "du": "Delhi University",
    "jnu": "JNU",
    "iitd": "IIT Delhi",
    "gk": "Greater Kailash",
    "gk1": "Greater Kailash",
    "gk2": "Greater Kailash",
    "gk-1": "Greater Kailash",
    "gk-2": "Greater Kailash",
    "hkv": "Hauz Khas",
    "nsp": "Netaji Subhash Place",
    "skk": "Sarai Kale Khan",
    "vk": "Vasant Kunj",
    "rg": "Rajouri Garden",
    "kb": "Karol Bagh",
    "mkt": "Majnu Ka Tilla",
    "south ex": "South Extension",
    "ina": "INA",
    "aiims": "AIIMS",
    "isbt": "Kashmere Gate",
    "gtb nagar": "GTB Nagar",
}

# Default locations file path (relative to this module)
_LOCATIONS_FILE = Path(__file__).parent / "locations.json"

# ---------------------------------------------------------------------------
# Regex patterns for extracting source / destination
# ---------------------------------------------------------------------------

# English: "from X to Y"
_PATTERN_FROM_TO = re.compile(
    r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s+(?:by|via|using|through|in)\b|[?.!,]|$)",
    re.IGNORECASE,
)

# Hindi: "X से Y तक" or "X se Y tak"
_PATTERN_HINDI_SE_TAK = re.compile(
    r"(.+?)\s+(?:से|se)\s+(.+?)\s+(?:तक|tak|तक|ko|को)(?:\s|[?.!,]|$)",
    re.IGNORECASE,
)

# Simple: "X to Y"
_PATTERN_X_TO_Y = re.compile(
    r"(?:^|[\s,])(.+?)\s+to\s+(.+?)(?:\s+(?:by|via|using|through|in|metro|bus|auto)\b|[?.!,]|$)",
    re.IGNORECASE,
)

# Hindi: "X से Y" (without tak)
_PATTERN_HINDI_SE = re.compile(
    r"(.+?)\s+(?:से|se)\s+(.+?)(?:\s+(?:by|via|metro|bus|auto|mein|में)\b|[?.!,]|$)",
    re.IGNORECASE,
)

# "near X" / "around X" / "X ke paas"
_PATTERN_NEAR = re.compile(
    r"\b(?:near|around|nearby|close to|ke paas|के पास|ke pass)\s+(.+?)(?:\s|[?.!,]|$)",
    re.IGNORECASE,
)

# "to X" (destination only)
_PATTERN_TO_DEST = re.compile(
    r"\b(?:to|towards|for|ke liye|के लिए)\s+(.+?)(?:\s+(?:by|via|using|through|in)\b|[?.!,]|$)",
    re.IGNORECASE,
)


class EntityExtractor:
    """Extracts source and destination locations from user queries.

    The extractor uses a multi-strategy approach:
    1. Regex pattern matching for structured phrases (``from X to Y``, etc.).
    2. Abbreviation expansion (``CP`` → ``Connaught Place``).
    3. Fuzzy string matching against a curated Delhi locations database.
    4. Support for Hindi (Devanagari) and Hinglish location names.

    Attributes:
        fuzzy_threshold: Minimum score (0-100) for fuzzy matching acceptance.
    """

    def __init__(
        self,
        locations_path: Optional[str | Path] = None,
        fuzzy_threshold: int = 75,
    ) -> None:
        """Initialise the entity extractor.

        Args:
            locations_path: Path to the ``locations.json`` file.  Defaults to
                ``src/classifier/locations.json``.
            fuzzy_threshold: Minimum ``rapidfuzz`` score (0-100) for a fuzzy
                match to be accepted.  Defaults to 75.
        """
        self.fuzzy_threshold: int = fuzzy_threshold

        # Internal data structures populated by _load_locations()
        self._locations: List[Dict[str, Any]] = []
        self._alias_to_canonical: Dict[str, str] = {}
        self._all_aliases: List[str] = []

        path = Path(locations_path) if locations_path else _LOCATIONS_FILE
        self._load_locations(path)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_locations(self, path: Path) -> None:
        """Load and index the locations database from a JSON file.

        Args:
            path: Absolute or relative path to ``locations.json``.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not path.exists():
            logger.error("Locations file not found: %s", path)
            raise FileNotFoundError(f"Locations file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        self._locations = data.get("locations", [])

        # Build alias → canonical lookup
        for loc in self._locations:
            canonical = loc["canonical"]
            # Map the canonical name itself
            self._alias_to_canonical[canonical.lower()] = canonical
            for alias in loc.get("aliases", []):
                self._alias_to_canonical[alias.lower()] = canonical

        self._all_aliases = list(self._alias_to_canonical.keys())

        # Merge the abbreviation map (lower priority – won't overwrite)
        for abbr, canonical in _ABBREVIATION_MAP.items():
            self._alias_to_canonical.setdefault(abbr.lower(), canonical)

        logger.info(
            "Loaded %d locations with %d aliases.",
            len(self._locations),
            len(self._all_aliases),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> Dict[str, Optional[str]]:
        """Extract source and destination locations from *text*.

        The method tries multiple pattern-matching strategies in order of
        specificity and returns the first successful extraction.

        Args:
            text: Raw user input string.

        Returns:
            A dictionary with keys:
                - ``source`` (*str | None*): Canonical name of the origin.
                - ``destination`` (*str | None*): Canonical name of the
                  destination.
        """
        if not text or not text.strip():
            return {"source": None, "destination": None}

        source: Optional[str] = None
        destination: Optional[str] = None

        # Strategy 1 – "from X to Y"
        source, destination = self._try_from_to(text)
        if source or destination:
            return {"source": source, "destination": destination}

        # Strategy 2 – Hindi "X से Y तक/को"
        source, destination = self._try_hindi_se_tak(text)
        if source or destination:
            return {"source": source, "destination": destination}

        # Strategy 3 – "X to Y" (simpler form)
        source, destination = self._try_x_to_y(text)
        if source or destination:
            return {"source": source, "destination": destination}

        # Strategy 4 – Hindi "X से Y"
        source, destination = self._try_hindi_se(text)
        if source or destination:
            return {"source": source, "destination": destination}

        # Strategy 5 – "near X" / "around X" (destination only)
        destination = self._try_near(text)
        if destination:
            return {"source": None, "destination": destination}

        # Strategy 6 – "to X" (destination only)
        destination = self._try_to_dest(text)
        if destination:
            return {"source": None, "destination": destination}

        # Strategy 7 – Scan all tokens for fuzzy location matches
        found = self._scan_for_locations(text)
        if len(found) >= 2:
            return {"source": found[0], "destination": found[1]}
        elif len(found) == 1:
            return {"source": None, "destination": found[0]}

        return {"source": None, "destination": None}

    # ------------------------------------------------------------------
    # Pattern-matching helpers
    # ------------------------------------------------------------------

    def _try_from_to(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Match ``from X to Y`` pattern."""
        match = _PATTERN_FROM_TO.search(text)
        if match:
            src = self._resolve_location(match.group(1).strip())
            dst = self._resolve_location(match.group(2).strip())
            if src or dst:
                return src, dst
        return None, None

    def _try_hindi_se_tak(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Match ``X से Y तक/को`` pattern."""
        match = _PATTERN_HINDI_SE_TAK.search(text)
        if match:
            src = self._resolve_location(match.group(1).strip())
            dst = self._resolve_location(match.group(2).strip())
            if src or dst:
                return src, dst
        return None, None

    def _try_x_to_y(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Match ``X to Y`` pattern."""
        match = _PATTERN_X_TO_Y.search(text)
        if match:
            src = self._resolve_location(match.group(1).strip())
            dst = self._resolve_location(match.group(2).strip())
            if src or dst:
                return src, dst
        return None, None

    def _try_hindi_se(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Match ``X से Y`` pattern."""
        match = _PATTERN_HINDI_SE.search(text)
        if match:
            src = self._resolve_location(match.group(1).strip())
            dst = self._resolve_location(match.group(2).strip())
            if src or dst:
                return src, dst
        return None, None

    def _try_near(self, text: str) -> Optional[str]:
        """Match ``near X`` / ``around X`` / ``X ke paas`` pattern."""
        match = _PATTERN_NEAR.search(text)
        if match:
            return self._resolve_location(match.group(1).strip())
        return None

    def _try_to_dest(self, text: str) -> Optional[str]:
        """Match ``to X`` / ``towards X`` pattern."""
        match = _PATTERN_TO_DEST.search(text)
        if match:
            return self._resolve_location(match.group(1).strip())
        return None

    # ------------------------------------------------------------------
    # Location resolution
    # ------------------------------------------------------------------

    def _resolve_location(self, raw: str) -> Optional[str]:
        """Resolve a raw text fragment to a canonical location name.

        Resolution order:
        1. Exact match (case-insensitive) against alias dictionary.
        2. Abbreviation map lookup.
        3. Fuzzy match via ``rapidfuzz`` (if available).

        Args:
            raw: Raw text fragment that might be a location.

        Returns:
            The canonical location name, or ``None`` if no match found.
        """
        if not raw:
            return None

        cleaned = raw.strip().rstrip("?.!,;:")

        # 1. Exact alias match
        canonical = self._alias_to_canonical.get(cleaned.lower())
        if canonical:
            return canonical

        # 2. Abbreviation map
        canonical = _ABBREVIATION_MAP.get(cleaned.lower())
        if canonical:
            return canonical

        # 3. Fuzzy matching
        return self._fuzzy_match(cleaned)

    def _fuzzy_match(self, text: str) -> Optional[str]:
        """Attempt fuzzy matching against all known location aliases.

        Args:
            text: Candidate location string.

        Returns:
            Canonical location name if a match above threshold is found,
            otherwise ``None``.
        """
        if not _HAS_RAPIDFUZZ:
            return self._exact_substring_match(text)

        if not self._all_aliases:
            return None

        result = process.extractOne(
            text.lower(),
            self._all_aliases,
            scorer=fuzz.WRatio,
            score_cutoff=self.fuzzy_threshold,
        )

        if result is not None:
            matched_alias, score, _ = result
            canonical = self._alias_to_canonical.get(matched_alias)
            logger.debug(
                "Fuzzy matched '%s' → '%s' (alias='%s', score=%.1f)",
                text,
                canonical,
                matched_alias,
                score,
            )
            return canonical

        return None

    def _exact_substring_match(self, text: str) -> Optional[str]:
        """Fallback exact substring matching when rapidfuzz is unavailable.

        Args:
            text: Candidate location string.

        Returns:
            Canonical location name if found, otherwise ``None``.
        """
        text_lower = text.lower()
        # Try to find text as a substring in aliases
        for alias, canonical in self._alias_to_canonical.items():
            if alias in text_lower or text_lower in alias:
                return canonical
        return None

    # ------------------------------------------------------------------
    # Token scanning
    # ------------------------------------------------------------------

    def _scan_for_locations(self, text: str) -> List[str]:
        """Scan the entire text for known locations via sliding window.

        Tries multi-word and single-word windows against the alias dictionary
        and fuzzy matcher.  Returns locations in order of appearance.

        Args:
            text: Full user input text.

        Returns:
            List of canonical location names found (deduplicated, ordered).
        """
        found: List[str] = []
        seen: set[str] = set()

        # Clean the text of common noise words for scanning
        words = text.split()
        n = len(words)

        # Try windows of decreasing size (up to 4 words)
        for window_size in range(min(4, n), 0, -1):
            for i in range(n - window_size + 1):
                candidate = " ".join(words[i : i + window_size])
                resolved = self._resolve_location(candidate)
                if resolved and resolved not in seen:
                    found.append(resolved)
                    seen.add(resolved)

        return found

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_location_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Look up full metadata for a location by name or alias.

        Args:
            name: Location name or alias.

        Returns:
            The full location dictionary from the database, or ``None``.
        """
        canonical = self._resolve_location(name)
        if not canonical:
            return None

        for loc in self._locations:
            if loc["canonical"] == canonical:
                return loc
        return None

    @property
    def location_count(self) -> int:
        """Return the number of locations in the database."""
        return len(self._locations)

    @property
    def alias_count(self) -> int:
        """Return the total number of aliases (including canonical names)."""
        return len(self._all_aliases)

    def __repr__(self) -> str:
        mode = "fuzzy" if _HAS_RAPIDFUZZ else "exact"
        return (
            f"EntityExtractor(locations={self.location_count}, "
            f"aliases={self.alias_count}, "
            f"mode={mode}, "
            f"threshold={self.fuzzy_threshold})"
        )
