"""Show, check and merge the annotations of regular eight-legged essays of 欽定四書文.

An essay is regular when its body has exactly four leg pairs (起股, 中股, 後股,
束股); see docs/baguwen_leg_alignment_method.md. Annotations are kept in
data/4_eight_legged_essays/qinding_regular/part_*.json, one file per annotator:

    {"essays": {"<index>": {"regular": true, "pairs": [<pair>, <pair>, <pair>, <pair>]},
                "<index>": {"regular": false, "reason": "..."}}}

Indices are into data/4_eight_legged_essays/qinding_sishuwen.json essays. A pair is either
[opening, closing words of leg A, opening, closing words of leg B] in the
unpunctuated text, read as in prepare_qinding_legs.py (the opening words of
leg A must occur only once in the essay), or {"a", "b"} with hand
punctuation, whose characters must match the text exactly.

    show <index> ...    print essays with their automatic punctuation
    check <file> ...    check annotation files and print the legs as cut
    merge               write data/4_eight_legged_essays/qinding_regular_annotations.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_qinding_legs import FRAGMENT_SPLIT, break_positions, cut

ESSAYS = Path("data/4_eight_legged_essays/qinding_sishuwen.json")
PARTS = Path("data/4_eight_legged_essays/qinding_regular")
MERGED = Path("data/4_eight_legged_essays/qinding_regular_annotations.json")


def locate(pair, text: str, breaks: set[int], start: int) -> tuple[list[str], list[str], int]:
    """Cut a pair from the text; return both legs as fragments and where leg B ends."""
    if isinstance(pair, dict):
        a = [f for f in FRAGMENT_SPLIT.split(pair["a"]) if f]
        b = [f for f in FRAGMENT_SPLIT.split(pair["b"]) if f]
        a_start = text.find("".join(a), start)
        b_start = text.find("".join(b), a_start + len("".join(a))) if a_start >= 0 else -1
        if a_start < 0 or b_start < 0:
            raise ValueError("hand punctuation does not match the text, or the pair starts before the previous one ends")
        return a, b, b_start + len("".join(b))
    open_a, close_a, open_b, close_b = pair
    if text.count(open_a) != 1:
        raise ValueError(f"opening words {open_a!r} occur {text.count(open_a)} times; give more characters")
    a_start = text.index(open_a)
    if a_start < start:
        raise ValueError("pair starts before the previous pair ends: legs must follow the text order 1A 1B 2A 2B ...; "
                         "parallel sentences inside one leg are not separate pairs")
    try:
        a_end = text.index(close_a, a_start) + len(close_a)
        b_start = text.index(open_b, a_end)
        b_end = text.index(close_b, b_start) + len(close_b)
    except ValueError:
        raise ValueError(f"words not found {pair}") from None
    return cut(text, breaks, a_start, a_end), cut(text, breaks, b_start, b_end), b_end


def check(entry: dict, essay: dict) -> tuple[list[str], list[str]]:
    """Problems of one essay's entry and a printout of its legs."""
    if not entry.get("regular"):
        return ([] if entry.get("reason") else ["not regular but no reason given"]), []
    pairs = entry.get("pairs", [])
    problems = [] if len(pairs) == 4 else [f"{len(pairs)} pairs instead of 4"]
    text, breaks = essay["text"], break_positions(essay["punctuated"])
    lines, end = [], 0
    for index, pair in enumerate(pairs):
        try:
            a, b, end = locate(pair, text, breaks, end)
        except ValueError as error:
            problems.append(f"pair {index + 1}: {error}")
            continue
        if len(a) != len(b):
            problems.append(f"pair {index + 1}: {len(a)} and {len(b)} fragments; give it with hand punctuation")
        lines += [f"  {index + 1}A ({len(a)}) {'，'.join(a)}", f"  {index + 1}B ({len(b)}) {'，'.join(b)}"]
    return problems, lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["show", "check", "merge"])
    parser.add_argument("targets", nargs="*")
    parser.add_argument("--quiet", action="store_true", help="With check, print problems only")
    args = parser.parse_args()
    essays = json.loads(ESSAYS.read_text(encoding="utf-8"))["essays"]

    if args.command == "show":
        for key in args.targets:
            essay = essays[int(key)]
            print(f"=== {key} {essay['period']} {essay['author']} 《{essay['title']}》\n{essay['punctuated']}\n")
        return

    files = [Path(t) for t in args.targets] if args.command == "check" else sorted(PARTS.glob("part_*.json"))
    merged, total = {}, 0
    for path in files:
        entries = json.loads(path.read_text(encoding="utf-8"))["essays"]
        for key, entry in sorted(entries.items(), key=lambda item: int(item[0])):
            problems, lines = check(entry, essays[int(key)])
            total += len(problems)
            if args.command == "check" and (problems or not args.quiet):
                status = "regular" if entry.get("regular") else f"not regular: {entry.get('reason', '')}"
                print(f"{key} {status}")
                print("\n".join([f"  !! {p}" for p in problems] + ([] if args.quiet else lines)))
            if key in merged:
                print(f"{key}: annotated twice ({path})")
                total += 1
            merged[key] = entry
    regular = sum(1 for e in merged.values() if e.get("regular"))
    print(f"{len(merged)} essays annotated, {regular} regular, {total} problems")
    if args.command == "merge":
        if total:
            raise SystemExit("nothing written")
        payload = {"note": "Regular eight-legged essays of 欽定四書文: every essay read; an essay is regular when its "
                           "body has exactly four leg pairs (起股, 中股, 後股, 束股). Keys are indices into "
                           "data/4_eight_legged_essays/qinding_sishuwen.json essays; pairs as in data/4_eight_legged_essays/qinding_leg_annotations.json; "
                           "an empty list means not regular (reasons in data/4_eight_legged_essays/qinding_regular/).",
                   "pairs": {k: (e["pairs"] if e.get("regular") else []) for k, e in
                             sorted(merged.items(), key=lambda item: int(item[0]))}}
        MERGED.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"Saved: {MERGED}")


if __name__ == "__main__":
    main()
