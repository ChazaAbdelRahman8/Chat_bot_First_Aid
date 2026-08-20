"""Deterministic page-text quality gate carried over from the v1 run."""

from __future__ import annotations

import re
from collections import Counter


MOJIBAKE_MARKERS = ("\ufffd", "\u25a1", "ï¿½")


def normalize_text(text: str) -> str:
    """Normalize extraction noise without changing the document wording."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def assess_text_quality(text: str, embedded_image_count: int) -> dict:
    """Assess the quality of extracted text and return metrics and reasons for failure."""
    stripped = text.strip()
    character_count = len(stripped)
    alphanumeric_count = sum(character.isalnum() for character in stripped)
    printable_count = sum(
        character.isprintable() or character in "\n\t" for character in stripped
    )
    tokens = re.findall(r"\b\w+\b", stripped)
    max_token_repeat_ratio = 0.0
    if tokens:
        max_token_repeat_ratio = (
            Counter(token.lower() for token in tokens).most_common(1)[0][1]
            / len(tokens)
        )

    metrics = {
        "character_count": character_count,
        "word_count": len(tokens),
        "alnum_ratio": round(alphanumeric_count / max(character_count, 1), 4),
        "printable_ratio": round(printable_count / max(character_count, 1), 4),
        "mojibake_count": sum(stripped.count(marker) for marker in MOJIBAKE_MARKERS),
        "max_token_repeat_ratio": round(max_token_repeat_ratio, 4),
        "embedded_image_count": embedded_image_count,
    }

    reasons: list[str] = []
    if character_count < 80:
        reasons.append("too_few_characters")
    if metrics["alnum_ratio"] < 0.45:
        reasons.append("low_alphanumeric_ratio")
    if metrics["printable_ratio"] < 0.97:
        reasons.append("low_printable_ratio")
    if metrics["mojibake_count"] > 2:
        reasons.append("encoding_artifacts")
    if max_token_repeat_ratio > 0.20 and len(tokens) >= 20:
        reasons.append("excessive_token_repetition")
    if embedded_image_count > 0 and character_count < 250:
        reasons.append("image_heavy_low_text")

    metrics["quality_pass"] = not reasons
    metrics["quality_reasons"] = reasons
    return metrics

