"""Check and collect the composed counterparts of the 欽定四書文 leg pairs.

data/4_eight_legged_essays/qinding_controls/composed.txt holds leg pairs written by hand, one block
per essay of data/4_eight_legged_essays/qinding_legs.json:

    @ <essay id> <topic of the composed pairs>
    A <member A of the essay's first pair>
    B <member B of the essay's first pair>
    A ...

The pairs of a block follow the essay's pairs in order and argue a different
saying of the Four Books (the topic). Each composed pair is checked against
the real pair it stands in for: the same number of fragments in both members,
every fragment within --tolerance characters of the corresponding real
fragment, every character one GujiBERT token, no fragment repeated
immediately within a member, and no run of --max-copy + 1 characters taken
over from the real essay. Lines starting with # are comments.

Writes data/4_eight_legged_essays/qinding_controls/composed.json in the format of
data/4_eight_legged_essays/qinding_legs.json, so that baguwen_pseudo_legs.py can read it; with
failures, reports them and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

from self_similarity_matrix import DEFAULT_MODEL

FRAGMENT_SPLIT = re.compile(r"[，。；：？！、]")


def fragments(text: str) -> list[str]:
    return [f for f in FRAGMENT_SPLIT.split(text) if f]


def parse(path: Path) -> list[dict]:
    blocks = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tag, _, rest = line.partition(" ")
        if tag == "@":
            essay_id, _, topic = rest.strip().partition(" ")
            blocks.append({"id": essay_id, "topic": topic.strip(), "members": []})
        elif tag in ("A", "B") and blocks:
            blocks[-1]["members"].append((tag, rest.strip(), number))
        else:
            raise SystemExit(f"{path}:{number}: cannot parse {line[:20]!r}")
    return blocks


def check_member(text: str, real: str, args, tokenizer, essay_text: str) -> list[str]:
    problems = []
    mine, theirs = fragments(text), fragments(real)
    if len(mine) != len(theirs):
        return [f"{len(mine)} fragments instead of {len(theirs)}"]
    gaps = [(i, len(x), len(y)) for i, (x, y) in enumerate(zip(mine, theirs)) if abs(len(x) - len(y)) > args.tolerance]
    if gaps:
        problems.append("lengths " + ", ".join(f"#{i + 1} {x} (real {y})" for i, x, y in gaps))
    if any(x == y for x, y in zip(mine, mine[1:])):
        problems.append("fragment repeated immediately")
    tokens = tokenizer(mine, add_special_tokens=False)["input_ids"]
    if any(len(t) != len(f) or tokenizer.unk_token_id in t for t, f in zip(tokens, mine)):
        bad = [c for f in mine for c in f if tokenizer.convert_tokens_to_ids(c) == tokenizer.unk_token_id]
        problems.append(f"tokenization ({''.join(bad) or 'merged characters'})")
    run = args.max_copy + 1
    copied = {f[i:i + run] for f in mine for i in range(len(f) - run + 1) if f[i:i + run] in essay_text}
    if copied:
        problems.append("copied " + " ".join(sorted(copied)))
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/4_eight_legged_essays/qinding_controls/composed.txt")
    parser.add_argument("--corpus", default="data/4_eight_legged_essays/qinding_legs.json")
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--max-copy", type=int, default=3,
                        help="Longest run of characters a composed fragment may share with the real essay")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="data/4_eight_legged_essays/qinding_controls/composed.json")
    args = parser.parse_args()

    essays = {e["id"]: e for e in json.loads(Path(args.corpus).read_text(encoding="utf-8"))["essays"]}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    failures, output_essays = [], []
    for block in parse(Path(args.source)):
        essay = essays.get(block["id"])
        if essay is None:
            failures.append(f"essay {block['id']}: not in {args.corpus}")
            continue
        members = block["members"]
        if [tag for tag, _, _ in members] != ["A", "B"] * len(essay["pairs"]):
            failures.append(f"essay {block['id']}: expected {len(essay['pairs'])} A/B pairs, got {''.join(t for t, _, _ in members)}")
            continue
        real_text = essay["text"] + "".join(p["a"] + p["b"] for p in essay["pairs"])
        pairs = []
        for index, real in enumerate(essay["pairs"]):
            (_, a, line_a), (_, b, line_b) = members[2 * index:2 * index + 2]
            for text, key, line in ((a, "a", line_a), (b, "b", line_b)):
                failures.extend(f"line {line} (essay {block['id']}, pair {index}, {key.upper()}): {p}"
                                for p in check_member(text, real[key], args, tokenizer, real_text))
            pairs.append({"a": a, "b": b, "fragments": len(fragments(a))})
        output_essays.append({"id": essay["id"], "template": essay["id"], "topic": block["topic"], "pairs": pairs})

    missing = sorted(set(essays) - {e["id"] for e in output_essays}, key=int)
    print(f"{len(output_essays)} essays, {sum(len(e['pairs']) for e in output_essays)} pairs composed; "
          f"{len(missing)} essays still missing" + (f" ({' '.join(missing[:10])} ...)" if missing else ""))
    if failures:
        print("\n".join(failures))
        print(f"{len(failures)} problems; nothing written")
        sys.exit(1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generator": "make_qinding_composed.py", "source": args.source, "template": args.corpus,
               "note": "Leg pairs written by hand after the hand-marked pairs of 欽定四書文, one topic per essay",
               "essays": output_essays}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
