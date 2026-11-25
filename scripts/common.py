"""Shared helpers for the Chinese repetition experiment scripts."""

from __future__ import annotations

import hashlib
import math
import re


INNER_POSITIONS = {2, 3}
OUTER_POSITIONS = {1, 4}
METERS = (5, 7)
METER_NAMES = {5: "penta", 7: "hepta"}
COUPLET_END_PUNCTUATION = re.compile(r"[。！？]")

# OCR/transcription placeholder markers ("missing character" boxes, the
# Unicode replacement character, etc.) that occasionally survive in the
# scraped source corpus. Private-use-area codepoints are font-specific OCR
# artifacts for the same reason. These are not real characters, so any line
# containing one should be dropped rather than scored.
JUNK_CHARACTERS = frozenset("□■○∴�")


def has_junk_characters(text: str) -> bool:
    """Return True if text contains an OCR placeholder or other junk character."""
    return any(ch in JUNK_CHARACTERS or 0xE000 <= ord(ch) <= 0xF8FF for ch in text)


def select_device(device_name: str = "auto"):
    """Choose the requested torch device, or the best available local default."""
    import torch

    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compact_text(text: str) -> str:
    """Remove whitespace from text for character-position analyses."""
    return "".join(text.split())


def split_couplet(couplet: str) -> tuple[str | None, str | None]:
    """Split a comma-separated couplet into two non-empty stripped lines."""
    lines = [line.strip() for line in couplet.split("，")]
    if len(lines) != 2 or not all(lines):
        return None, None
    return lines[0], lines[1]


def poem_id_for(meter: int, content: str) -> str:
    """Return a stable id for one regulated poem.

    Identity is the meter plus the normalized poem text, so the same poem
    filed twice cannot be split across train and test. Title and author are
    not part of the id: anthologies reprint the same text under variant headings.
    """
    normalized = compact_text(content)
    if "评注" in normalized:
        normalized = normalized.split("评注")[0]
    payload = f"{meter}\n{normalized}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def parse_regulated_couplets(poem_text: str, line_length: int) -> list[tuple[str, int]]:
    """Parse a regulated-poem (律詩) text into its first four couplets.

    A couplet is kept only if both of its lines have exactly `line_length`
    characters (5 for pentasyllabic, 7 for heptasyllabic). Couplets are
    delimited by any of 。！？, since the final couplet of a poem sometimes
    ends in ！ or ？ instead of the usual 。.
    """
    if "评注" in poem_text:
        poem_text = poem_text.split("评注")[0]

    couplets = []
    for part in COUPLET_END_PUNCTUATION.split(poem_text):
        part = part.strip()
        if not part or "，" not in part:
            continue

        line1, line2 = split_couplet(part)
        if (
            line1 is not None
            and line2 is not None
            and len(line1) == line_length
            and len(line2) == line_length
            and not has_junk_characters(line1)
            and not has_junk_characters(line2)
        ):
            couplets.append(f"{line1}，{line2}")

    return [(couplet, idx + 1) for idx, couplet in enumerate(couplets[:4])]


def io_label_for_position(position: int) -> str:
    """Return the structural inner/outer label for a regulated-poem couplet."""
    if position in INNER_POSITIONS:
        return "i"
    if position in OUTER_POSITIONS:
        return "o"
    raise ValueError(f"Unsupported couplet position: {position}")


def ensure_padding_token(tokenizer) -> None:
    """Ensure padding works for GPT-style tokenizers that do not define PAD."""
    if tokenizer.pad_token is not None:
        return

    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def format_optional_float(value: float | None, digits: int = 4) -> str:
    """Format floats while keeping missing statistics readable."""
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"
