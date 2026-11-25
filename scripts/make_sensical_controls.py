"""Compose meaningful passages that follow the Mengzi 6A10 template.

Frames, verb pairs, and content pairs are drawn independently from a slot bank
(data/1_mengzi_prose/surprisal_controls/sensical_bank.json). Each slot list is dealt from a
shuffled deck that is reshuffled only when exhausted, so fillers are used as
evenly as possible. A combination is rejected if a content character also
occurs in its frame or if it does not match the template's length and
punctuation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DEFAULT_BANK = "data/1_mengzi_prose/surprisal_controls/sensical_bank.json"
DEFAULT_TEMPLATE = "data/1_mengzi_prose/mengzi_surprisal.txt"
DEFAULT_OUTPUT = "data/1_mengzi_prose/surprisal_controls/sensical.json"
PUNCTUATION = set("，；。")


class Deck:
    """Deal items in shuffled order, reshuffling only once all have been dealt."""

    def __init__(self, items: list, rng: random.Random) -> None:
        self.items = list(items)
        self.rng = rng
        self.pile: list = []

    def deal(self):
        if not self.pile:
            self.pile = self.items[:]
            self.rng.shuffle(self.pile)
        return self.pile.pop()


def compose(decks: dict[str, Deck], template: str) -> tuple[str, dict[str, str]]:
    a, bb = decks["goods"].deal()
    c, d = decks["values"].deal()
    v1, v2 = decks["verbs"].deal()
    slots = {
        "A": a,
        "BB": bb,
        "C": c,
        "D": d,
        "V1": v1,
        "V2": v2,
        "F1": decks["F1"].deal(),
        "F2": decks["F2"].deal(),
    }
    return template.format(**slots), slots


def is_valid(text: str, slots: dict[str, str], template: str) -> bool:
    if len(text) != len(template):
        return False
    if any((a in PUNCTUATION) != (b in PUNCTUATION) for a, b in zip(text, template)):
        return False
    content = set(slots["A"] + slots["BB"] + slots["C"] + slots["D"])
    frame = set(slots["F1"] + slots["F2"] + slots["V1"] + slots["V2"] + "亦而者也")
    return not content & frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default=DEFAULT_BANK)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    bank = json.loads(Path(args.bank).read_text(encoding="utf-8"))
    template = Path(args.template).read_text(encoding="utf-8").strip()
    rng = random.Random(args.seed)
    decks = {key: Deck(bank[key], rng) for key in ("goods", "values", "verbs", "F1", "F2")}
    passages: list[dict[str, object]] = []
    seen: set[str] = set()
    while len(passages) < args.count:
        text, slots = compose(decks, bank["template"])
        if text in seen or not is_valid(text, slots, template):
            continue
        seen.add(text)
        passages.append({"id": f"sensical_{len(passages):02d}", "text": text, "slots": slots})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generator": "make_sensical_controls.py", "seed": args.seed, "passages": passages}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for passage in passages:
        print(passage["text"])


if __name__ == "__main__":
    main()
