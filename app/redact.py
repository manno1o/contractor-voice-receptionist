"""Strip PII from transcripts before they get stored or sent anywhere."""
from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("SSN", re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")),
    ("PHONE", re.compile(r"(?:\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")),
    (
        "ADDRESS",
        re.compile(
            r"\b\d{1,6}\s+(?:[NSEW]\.?\s+)?(?:[A-Z][a-z]+\s){1,3}"
            r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court|Pl|Place|Way|Pkwy|Parkway)\b\.?",
        ),
    ),
    ("ZIP", re.compile(r"\b\d{5}(?:-\d{4})?\b")),
]

# Spoken digit sequences produced by STT ("five five five one two three ...").
_DIGIT_WORDS = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)"
_SPOKEN_NUMBER = re.compile(rf"\b{_DIGIT_WORDS}(?:[\s,-]+{_DIGIT_WORDS}){{6,}}\b", re.IGNORECASE)


def redact(text: str) -> str:
    if not text:
        return text
    out = _SPOKEN_NUMBER.sub("[PHONE]", text)
    for label, pattern in _PATTERNS:
        out = pattern.sub(f"[{label}]", out)
    return out


def redact_transcript(turns: list[dict]) -> list[dict]:
    return [{**t, "text": redact(t.get("text", ""))} for t in turns]
