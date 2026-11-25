"""Training corpora in which the inner couplets are replaced by degraded couplets.

Reads data/2-3_regulated_verse/couplets_{meter}_train.json and writes
data/2-3_regulated_verse/degraded_train/{meter}_{condition}.json in the same format, so that
finetune_io_gpt.py (--data-path) reads it unchanged. Only the inner couplets
(label "i") are degraded; the outer couplets (label "o") are kept as they are,
so that the control token still marks the regime. Every degraded couplet is
built as the held-out control couplets of make_couplet_controls.py are, with
donors taken from the inner couplets of the training set only:

- recombined: each segment of the line's rhythm (2+3 characters in
  pentasyllabic, 2+2+3 in heptasyllabic lines) is taken, in both lines at
  once, from another inner couplet, a different one for every segment
  (counterpart of the segswap test set);
- wordswap: the same with the word units of the line (2+2+1 characters in
  pentasyllabic, 2+2+2+1 in heptasyllabic lines; counterpart of the wordswap
  test set);
- nonsense: every pair of corresponding characters is taken from the same
  position of another inner couplet, a different one for every position
  (counterpart of the pairswap test set);
- shuffled: the characters of the second line are permuted (counterpart of
  the charshuf test set).

Couplet i of the corpus uses the string seed "{--seed}-{i}". With --limit, only the first N
couplets of the corpus are used (smoke tests).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from common import split_couplet
from make_couplet_controls import SEGMENTS, WORDS, shuffle_line

CONDITIONS = ("recombined", "wordswap", "nonsense", "shuffled")
LENGTHS = {"penta": 5, "hepta": 7}


def degrade(item: dict, condition: str, meter: str, donors: list[tuple[str, str]], rng: random.Random) -> dict:
    line1, line2 = split_couplet(item["couplet"])
    if condition == "shuffled":
        line2 = shuffle_line(line2, rng)
    else:
        spans = {"recombined": SEGMENTS[meter], "wordswap": WORDS[meter]}.get(
            condition, [(k, k + 1) for k in range(LENGTHS[meter])])
        picked = []
        while len(picked) < len(spans):
            donor = donors[rng.randrange(len(donors))]
            if donor != (line1, line2) and donor not in picked:
                picked.append(donor)
        line1 = "".join(d[0][a:b] for (a, b), d in zip(spans, picked))
        line2 = "".join(d[1][a:b] for (a, b), d in zip(spans, picked))
    return {**item, "couplet": f"{line1}，{line2}", "condition": condition, "source_couplet": item["couplet"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meter", choices=list(LENGTHS), required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N couplets (smoke tests)")
    parser.add_argument("--output-dir", default="data/2-3_regulated_verse/degraded_train")
    args = parser.parse_args()

    corpus = json.loads(Path(f"data/2-3_regulated_verse/couplets_{args.meter}_train.json").read_text(encoding="utf-8"))
    if args.limit:
        corpus = corpus[:args.limit]
    length = LENGTHS[args.meter]
    pairs = [split_couplet(item["couplet"]) for item in corpus]
    bad = [item["couplet"] for item, (a, b) in zip(corpus, pairs) if a is None or len(a) != length or len(b) != length]
    if bad:
        raise SystemExit(f"{len(bad)} couplets without two lines of {length} characters, e.g. {bad[:3]}")
    donors = [pair for item, pair in zip(corpus, pairs) if item["label"] == "i"]

    out = []
    for index, item in enumerate(corpus):
        if item["label"] == "i":
            # A string seed is hashed (SHA-512), which keeps the draws of neighbouring couplets independent;
            # consecutive integer seeds give correlated first draws from randrange.
            out.append(degrade(item, args.condition, args.meter, donors, random.Random(f"{args.seed}-{index}")))
        else:
            out.append(item)

    changed = sum(o["couplet"] != i["couplet"] for o, i in zip(out, corpus))
    labels = Counter(o["label"] for o in out)
    lengths = Counter(len(line) for o in out for line in split_couplet(o["couplet"]))
    output = Path(args.output_dir) / f"{args.meter}_{args.condition}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{args.meter} {args.condition}: {len(out)} couplets, labels {dict(labels)}, line lengths {dict(lengths)}, "
          f"{changed} inner couplets changed: {output}")
    for o in out[:6]:
        if o["label"] == "i":
            print(f"  {o['source_couplet']}  ->  {o['couplet']}")


if __name__ == "__main__":
    main()
