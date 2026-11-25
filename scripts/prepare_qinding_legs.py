"""Extract the hand-marked leg pairs of 欽定四書文 and cut them into fragments.

Reads data/4_eight_legged_essays/qinding_leg_annotations.json (see its note) and
data/4_eight_legged_essays/qinding_sishuwen.json. A pair given by opening and closing words is cut
from the essay's unpunctuated text, and each member is divided into fragments
where the essay's automatic punctuation puts a break. If the two members then
have different numbers of fragments, the pair is reported and left out until
it is given with hand punctuation ({"a", "b"}), whose characters must match
the essay text exactly.

Only leg pairs count, not single antithetical sentences: a pair is kept when
both members have at least --min-fragments fragments and --min-chars
characters, and an essay is kept when it has at least --min-pairs such pairs.

An essay given with --original (id=path) takes its pairs instead from an
edited text with <leg> tags, all partner legs (1-2, 3-4, 5-6, 7-8) with that
text's punctuation and no length limits: by default Chen Zilong's essay
(no. 346), whose middle legs would otherwise fall under --min-fragments and
--min-chars, so that it enters the corpus with the four pairs analysed
elsewhere in the project.

Writes data/4_eight_legged_essays/qinding_legs.json in the format of data/4_eight_legged_essays/baguwen_corpus.json, so
that baguwen_pseudo_legs.py can read it with --corpus.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from baguwen_fragment_space import PAIRS, leg_fragments
from prepare_qinding_sishuwen import BREAKS, HAN

FRAGMENT_SPLIT = re.compile(r"[，。；：？！、]")


def break_positions(punctuated: str) -> set[int]:
    """Character positions (in the unpunctuated text) after which the punctuation breaks."""
    positions, count = set(), 0
    for char in punctuated:
        if char in BREAKS:
            positions.add(count)
        elif HAN.match(char):
            count += 1
    return positions


def cut(text: str, breaks: set[int], start: int, end: int) -> list[str]:
    fragments, current = [], ""
    for position in range(start, end):
        current += text[position]
        if position + 1 in breaks and current:
            fragments.append(current)
            current = ""
    if current:
        fragments.append(current)
    return fragments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/4_eight_legged_essays/qinding_leg_annotations.json"))
    parser.add_argument("--essays", type=Path, default=Path("data/4_eight_legged_essays/qinding_sishuwen.json"))
    parser.add_argument("--min-fragments", type=int, default=4)
    parser.add_argument("--min-chars", type=int, default=25, help="Characters per member of a leg pair")
    parser.add_argument("--min-pairs", type=int, default=2, help="Leg pairs an essay needs to be kept")
    parser.add_argument("--original", nargs="*", default=["346=data/4_eight_legged_essays/baguwen_chen_zilong.txt"],
                        help="id=path pairs of essays whose leg pairs come from an edited text with <leg> tags")
    parser.add_argument("--output", type=Path, default=Path("data/4_eight_legged_essays/qinding_legs.json"))
    args = parser.parse_args()
    originals = dict(spec.split("=", 1) for spec in args.original)

    essays = json.loads(args.essays.read_text(encoding="utf-8"))["essays"]
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    out, problems, kept = [], 0, 0
    for key, marked in annotations["pairs"].items():
        essay = essays[int(key)]
        text, breaks = essay["text"], break_positions(essay["punctuated"])
        pairs = []
        if key in originals:
            legs = leg_fragments(Path(originals[key]).read_text(encoding="utf-8"))
            pairs = [{"a": "，".join(legs[a]), "b": "，".join(legs[b]), "fragments": len(legs[a])} for a, b in PAIRS]
            marked = []
        for index, pair in enumerate(marked):
            if isinstance(pair, dict):
                a = [f for f in FRAGMENT_SPLIT.split(pair["a"]) if f]
                b = [f for f in FRAGMENT_SPLIT.split(pair["b"]) if f]
                if "".join(a) not in text or "".join(b) not in text:
                    print(f"{key} pair {index}: hand punctuation does not match the text")
                    problems += 1
                    continue
            else:
                open_a, close_a, open_b, close_b = pair
                try:
                    a_start = text.index(open_a)
                    a_end = text.index(close_a, a_start) + len(close_a)
                    b_start = text.index(open_b, a_end)
                    b_end = text.index(close_b, b_start) + len(close_b)
                except ValueError:
                    print(f"{key} pair {index}: words not found {pair}")
                    problems += 1
                    continue
                a, b = cut(text, breaks, a_start, a_end), cut(text, breaks, b_start, b_end)
            if len(a) != len(b):
                print(f"{key} pair {index}: {len(a)} and {len(b)} fragments\n   A {'，'.join(a)}\n   B {'，'.join(b)}")
                problems += 1
                continue
            if len(a) < args.min_fragments or min(len("".join(a)), len("".join(b))) < args.min_chars:
                continue
            pairs.append({"a": "，".join(a), "b": "，".join(b), "fragments": len(a)})
        if len(pairs) < args.min_pairs:
            continue
        kept += len(pairs)
        out.append({"id": key, "title": essay["title"], "author": essay["author"], "date": essay["period"],
                    "url": "", "text": text, "pairs": pairs})
    args.output.write_text(json.dumps({"source": "欽定四書文, legs marked by hand", "essays": out},
                                      ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(out)} essays with at least {args.min_pairs} leg pairs, {kept} pairs kept, {problems} pairs to fix")


if __name__ == "__main__":
    main()
