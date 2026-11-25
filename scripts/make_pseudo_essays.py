"""Pseudo-essays of non-parallel prose matched to the regular eight-legged essays.

For every essay of --corpus (four leg pairs each), a run of consecutive
fragments of one piece of --prose (古文觀止) is cut into eight pseudo-legs with
the fragment counts of the essay's eight legs, in text order (1A, 1B, 2A, ...).
Every fragment lies within the range of the real fragments' lengths, and no
pseudo-leg repeats a fragment immediately (a zero transition has no
direction). Pieces are drawn at random (--seed), each at most once while
unused pieces with a long enough run remain.

The pseudo-essays have the essays' numbers of pairs and fragments, so that
the same prose serves as the comparison for the pair-level measure
(baguwen_pseudo_legs.py, which reads pseudo-pairs 1A-1B, 2A-2B, ...) and the
leg-level measure (baguwen_leg_streams.py, which reads the eight legs).
Writes them in the format of data/4_eight_legged_essays/qinding_legs.json, with the id of the
essay each one is matched to.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from baguwen_fragment_space import FRAGMENT_SPLIT
from baguwen_pseudo_legs import QUOTES


def split(text: str) -> list[str]:
    return [f for f in FRAGMENT_SPLIT.split(text) if f]


def runs(pieces: list[dict], min_chars: int, max_chars: int) -> list[tuple[str, list[str]]]:
    """Maximal runs of consecutive fragments within the length range, per piece."""
    out = []
    for piece in pieces:
        run = []
        for f in split(QUOTES.sub("", piece["text"])) + [""]:
            if min_chars <= len(f) <= max_chars:
                run.append(f)
            else:
                if run:
                    out.append((piece["title"], run))
                run = []
    return out


def cut(window: list[str], counts: list[int]) -> list[list[str]] | None:
    legs = []
    for count in counts:
        legs.append(window[:count])
        window = window[count:]
    return None if any(x == y for leg in legs for x, y in zip(leg, leg[1:])) else legs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/4_eight_legged_essays/qinding_regular100/legs.json"))
    parser.add_argument("--prose", type=Path, default=Path("data/4_eight_legged_essays/pseudo_legs/guwen.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/4_eight_legged_essays/qinding_regular100/guwen_pseudo.json"))
    args = parser.parse_args()

    essays = json.loads(args.corpus.read_text(encoding="utf-8"))["essays"]
    lengths = [len(f) for e in essays for p in e["pairs"] for key in ("a", "b") for f in split(p[key])]
    min_chars, max_chars = min(lengths), max(lengths)
    available = runs(json.loads(args.prose.read_text(encoding="utf-8"))["pieces"], min_chars, max_chars)
    rng, used, out = random.Random(args.seed), set(), []
    for essay in essays:
        counts = [len(split(p[key])) for p in essay["pairs"] for key in ("a", "b")]
        total = sum(counts)
        fitting = [(title, run) for title, run in available if len(run) >= total]
        fresh = [(title, run) for title, run in fitting if title not in used]
        for attempt in range(1000):
            title, run = rng.choice(fresh or fitting)
            start = rng.randrange(len(run) - total + 1)
            legs = cut(run[start:start + total], counts)
            if legs:
                break
        else:
            raise SystemExit(f"essay {essay['id']}: no window without repeated fragments")
        used.add(title)
        out.append({"id": essay["id"], "template": essay["id"], "title": title, "start": start,
                    "pairs": [{"a": "，".join(legs[2 * i]), "b": "，".join(legs[2 * i + 1]), "fragments": counts[2 * i]}
                              for i in range(len(legs) // 2)]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generator": "make_pseudo_essays.py", "prose": str(args.prose), "template": str(args.corpus),
               "min_chars": min_chars, "max_chars": max_chars, "seed": args.seed, "essays": out}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(out)} pseudo-essays from {len(used)} pieces, fragments of {min_chars}-{max_chars} characters: {args.output}")


if __name__ == "__main__":
    main()
