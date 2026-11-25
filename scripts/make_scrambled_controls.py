"""Completely scramble a passage: shuffle all of its characters, with no structure kept.

Every character of the template (punctuation included) is shuffled across the
whole passage, so each passage has the template's character inventory and
length but neither its order nor its pattern of repetition. Line breaks and
<leg> tags stay where they are, so section lengths match the template and
section boundaries can still be drawn at the same positions. Passage i uses
seed --seed + i.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

DEFAULT_TEMPLATE = "data/1_mengzi_prose/mengzi_surprisal.txt"
DEFAULT_OUTPUT = "data/1_mengzi_prose/surprisal_controls/scrambled.json"
FIXED = re.compile(r"</?leg>|\s")


def scramble(template: str, rng: random.Random) -> str:
    pieces = [piece for piece in re.split(r"(</?leg>|\s)", template) if piece]
    chars = [char for piece in pieces if not FIXED.fullmatch(piece) for char in piece]
    rng.shuffle(chars)
    shuffled = iter(chars)
    return "".join(piece if FIXED.fullmatch(piece) else "".join(next(shuffled) for _ in piece) for piece in pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    template = Path(args.template).read_text(encoding="utf-8").strip()
    passages = []
    for index in range(args.count):
        seed = args.seed + index
        passages.append({"id": f"scrambled_{index:02d}", "seed": seed, "text": scramble(template, random.Random(seed))})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generator": "make_scrambled_controls.py", "template": args.template, "passages": passages}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for passage in passages[:3]:
        print(passage["text"])


if __name__ == "__main__":
    main()
