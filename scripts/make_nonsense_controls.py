"""Fill a passage's repetition template with random, frequency-matched characters.

Every distinct character in the template passage is replaced by a different
character of similar frequency in the full Mengzi (data/1_mengzi_prose/mengzi_full.json, from
the chinese-poetry dataset): for each template character, one of the
--neighbors characters nearest to it in log frequency is drawn at random.
Template characters are matched from most to least frequent, and no character
is used twice in a passage. The pool excludes all characters of the template,
any character outside the model vocabulary, and any character whose simplified
form (OpenCC t2s, as read by the Qwen models) collides with another choice, so
the repetition pattern survives conversion. Matching frequency keeps character
rarity from inflating surprisal. Repeated characters stay repeated and
punctuation is kept, so each passage has the same length, repetition structure,
and approximate character frequencies as the template but no meaning. Passage i
uses seed --seed + i.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import opencc
from transformers import AutoTokenizer

from lm_scoring_utils import DEFAULT_MODEL


DEFAULT_TEMPLATE = "data/1_mengzi_prose/mengzi_surprisal.txt"
DEFAULT_POOL_JSON = "data/1_mengzi_prose/mengzi_full.json"
DEFAULT_OUTPUT = "data/1_mengzi_prose/surprisal_controls/nonsense.json"
PUNCTUATION = set("，；。、！？：「」『』")


def char_counts(json_path: Path) -> Counter:
    chapters = json.loads(json_path.read_text(encoding="utf-8"))
    return Counter(
        char
        for chapter in chapters
        for paragraph in chapter["paragraphs"]
        for char in paragraph
        if "一" <= char <= "鿿"
    )


def matched_replacements(template_chars, counts, pool, neighbors, rng, converter) -> dict[str, str]:
    """Map each template character to a random near neighbor in log frequency, without reuse."""
    mapping, used_simplified = {}, set()
    # Shuffle before the stable sort so that characters of equal frequency are ranked at random.
    shuffled = rng.sample(pool, len(pool))
    for char in sorted(template_chars, key=lambda c: (-counts[c], c)):
        target = math.log(counts[char])
        candidates = sorted(
            (c for c in shuffled if converter.convert(c) not in used_simplified),
            key=lambda c: abs(math.log(counts[c]) - target),
        )[:neighbors]
        choice = rng.choice(candidates)
        mapping[char] = choice
        used_simplified.add(converter.convert(choice))
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--pool-json", default=DEFAULT_POOL_JSON)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    template = Path(args.template).read_text(encoding="utf-8").strip()
    # Only CJK characters are replaced, so punctuation, line breaks, and <leg> tags pass through.
    template_chars = list(dict.fromkeys(c for c in template if c not in PUNCTUATION and "一" <= c <= "鿿"))
    vocab = AutoTokenizer.from_pretrained(args.model).get_vocab()
    converter = opencc.OpenCC("t2s")
    counts = char_counts(Path(args.pool_json))
    template_simplified = {converter.convert(c) for c in template_chars}
    pool = sorted(
        c for c in counts
        if c not in template_chars and c in vocab and converter.convert(c) not in template_simplified
    )

    passages = []
    for index in range(args.count):
        seed = args.seed + index
        mapping = matched_replacements(template_chars, counts, pool, args.neighbors, random.Random(seed), converter)
        text = "".join(mapping.get(char, char) for char in template)
        passages.append({"id": f"nonsense_{index:02d}", "seed": seed, "text": text, "mapping": mapping})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generator": "make_nonsense_controls.py",
        "pool_json": args.pool_json,
        "matching": f"random choice among the {args.neighbors} nearest characters in log frequency",
        "pool_size": len(pool),
        "passages": passages,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Pool: {len(pool)} characters")
    for passage in passages:
        print(passage["text"])


if __name__ == "__main__":
    main()
