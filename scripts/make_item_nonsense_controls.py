"""Keep a passage's frame and replace only its items with random, frequency-matched characters.

The items of the Mengzi 6A10 template are the characters at positions 1, 8-9,
33, and 41 (魚, 熊掌, 生, 義), which recur in the two conclusions (25, 28-29,
57, 60). For every composed passage (data/1_mengzi_prose/surprisal_controls/sensical.json),
and for the Mengzi passage itself under seeds --seed + i, these item characters
are replaced as in make_nonsense_controls.py: each by a random choice among the
--neighbors characters of the full Mengzi nearest to it in log frequency, with
no reuse, no character already in the passage, and no collision after OpenCC
t2s. Items absent from the Mengzi are matched as if they occurred once. Every
frame character stays in place, so each passage keeps its grammar and
repetition but its items no longer mean anything.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import opencc
from transformers import AutoTokenizer

from lm_scoring_utils import DEFAULT_MODEL
from make_nonsense_controls import DEFAULT_POOL_JSON, DEFAULT_TEMPLATE, char_counts, matched_replacements


ITEM_POSITIONS = [0, 7, 8, 32, 40]  # 0-based first occurrences of the items
DEFAULT_COMPOSED = "data/1_mengzi_prose/surprisal_controls/sensical.json"
DEFAULT_OUTPUT_DIR = "data/1_mengzi_prose/surprisal_controls"


def replace_items(text, counts, vocab, neighbors, rng, converter) -> tuple[str, dict[str, str]]:
    items = list(dict.fromkeys(text[i] for i in ITEM_POSITIONS))
    taken = {converter.convert(c) for c in text}
    pool = sorted(c for c in counts if c in vocab and c not in text and converter.convert(c) not in taken)
    floored = {**{c: 1 for c in items}, **{c: n for c, n in counts.items() if n > 0}}
    mapping = matched_replacements(items, floored, pool, neighbors, rng, converter)
    return "".join(mapping.get(c, c) for c in text), mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composed", default=DEFAULT_COMPOSED)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--pool-json", default=DEFAULT_POOL_JSON)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    counts = char_counts(Path(args.pool_json))
    vocab = AutoTokenizer.from_pretrained(args.model).get_vocab()
    converter = opencc.OpenCC("t2s")
    composed = json.loads(Path(args.composed).read_text(encoding="utf-8"))["passages"]
    template = Path(args.template).read_text(encoding="utf-8").strip()
    sources = {
        "items_random_composed_frame": [(p["id"], p["text"]) for p in composed],
        "items_random_mengzi_frame": [(f"mengzi_{i:02d}", template) for i in range(args.count)],
    }

    for name, items in sources.items():
        passages = []
        for index, (source_id, text) in enumerate(items):
            seed = args.seed + index
            new_text, mapping = replace_items(text, counts, vocab, args.neighbors, random.Random(seed), converter)
            passages.append({"id": f"{name}_{index:02d}", "source": source_id, "seed": seed,
                             "text": new_text, "mapping": mapping})
        output = Path(args.output_dir) / f"{name}.json"
        payload = {"generator": "make_item_nonsense_controls.py", "pool_json": args.pool_json,
                   "matching": f"random choice among the {args.neighbors} nearest characters in log frequency",
                   "passages": passages}
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{output}: {len(passages)} passages, e.g.\n  {passages[0]['text']}\n  {passages[1]['text']}")


if __name__ == "__main__":
    main()
