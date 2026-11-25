"""Couplet corpora encoded with a global cipher, at the level of the character or of the word.

The cipher keeps every structure of the corpus (line length, rhythm, the
recurrence of units within and across couplets and poems, the correspondence
of the two lines, the inner/outer labels) and removes the sense that the
pretrained model knows. It is applied to the whole training corpus
(data/2-3_regulated_verse/couplets_{meter}_train.json, inner and outer couplets alike) and to the
held-out real inner and outer couplets of make_couplet_controls.py
(data/2-3_regulated_verse/couplet_controls/{meter}_{inner,outer}.json), with one mapping, after
conversion to traditional characters with OpenCC (as finetune_io_gpt.py
converts its training data), so that the model is fine-tuned with
--no-convert and scored without conversion.

- char: every character is replaced, wherever it occurs, by another character
  of similar frequency: the characters are ranked by their frequency in the
  training corpus and permuted at random within bands of --band consecutive
  ranks, without fixed points. Characters that recur stay recurring, so that
  every word, and every character shared by different words, keeps its
  pattern under a new, mostly meaningless form.
- word: every line is cut into word units (2+2+1 characters in pentasyllabic,
  2+2+2+1 in heptasyllabic lines, reading the final three-character segment as
  2+1), and every distinct unit is replaced, wherever it occurs, by its own
  random pseudo-unit of the same length, whose characters are drawn according
  to their frequency in the training corpus (distinct units always receive
  distinct pseudo-units). Units keep their identity and their recurrence, while
  a character shared by different units no longer is.

Only characters that GujiGPT's tokenizer reads as single known tokens are
mapped or drawn; others are kept and counted. Writes, under --output-dir,
{meter}_{level}cipher_train.json (training format), {meter}_{level}cipher_inner.json
and {meter}_{level}cipher_outer.json (test sets, with "couplet" already in
traditional characters) and {meter}_{level}cipher_map.json (the mapping).
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path

import opencc
from transformers import AutoTokenizer

from make_couplet_controls import WORDS

LENGTHS = {"penta": 5, "hepta": 7}
MODEL = "hsc748NLP/GujiGPT_fan"


def is_han(char: str) -> bool:
    return "一" <= char <= "鿿" or "㐀" <= char <= "䶿"


def lines_of(couplet: str) -> tuple[str, str]:
    first, second = couplet.split("，")
    return first, second


def char_cipher(train_lines: list[str], all_lines: list[str], ok, band: int, rng: random.Random) -> dict[str, str]:
    counts = Counter(c for line in train_lines for c in line)
    for line in all_lines:  # characters seen only in the test sets go to the rare end
        for c in line:
            counts.setdefault(c, 0)
    ranked = [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if ok(c)]
    mapping = {}
    for start in range(0, len(ranked), band):
        chars = ranked[start:start + band]
        if len(chars) < 2:
            mapping.update({c: c for c in chars})
            continue
        shuffled = chars[:]
        rng.shuffle(shuffled)
        # A cyclic shift of a random order is a permutation without fixed points.
        mapping.update({c: shuffled[(shuffled.index(c) + 1) % len(shuffled)] for c in chars})
    return mapping


def word_units(line: str, meter: str) -> list[str]:
    return [line[a:b] for a, b in WORDS[meter]]


def word_cipher(train_lines: list[str], all_lines: list[str], meter: str, ok, rng: random.Random) -> dict[str, str]:
    # Every usable character of the corpus can be drawn, weighted by its training frequency (+1, so that
    # characters seen only in the test sets can be drawn too).
    char_counts = Counter(c for line in train_lines for c in line if ok(c))
    for line in all_lines:
        for c in line:
            if ok(c):
                char_counts[c] += 0
    pool, weights = zip(*((c, n + 1) for c, n in char_counts.items()))
    cum_weights = list(itertools.accumulate(weights))
    units = list(dict.fromkeys(u for line in all_lines for u in word_units(line, meter)))
    for size in {len(u) for u in units}:
        if sum(len(u) == size for u in units) > 0.9 * len(pool) ** size:
            raise SystemExit(f"Too few characters ({len(pool)}) for distinct pseudo-units of length {size}")
    rng.shuffle(units)
    mapping, used = {}, set()
    for unit in units:
        while True:
            candidate = "".join(rng.choices(pool, cum_weights=cum_weights, k=len(unit)))
            if candidate not in used:
                break
        used.add(candidate)
        mapping[unit] = candidate
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meter", choices=list(LENGTHS), required=True)
    parser.add_argument("--level", choices=["char", "word"], required=True)
    parser.add_argument("--band", type=int, default=10, help="char level: ranks per frequency band")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N training couplets (smoke tests)")
    parser.add_argument("--output-dir", default="data/2-3_regulated_verse/cipher")
    args = parser.parse_args()

    to_traditional = opencc.OpenCC("s2t")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    @lru_cache(maxsize=None)
    def ok(c: str) -> bool:
        return is_han(c) and len(tokenizer.tokenize(c)) == 1 and tokenizer.convert_tokens_to_ids(c) != tokenizer.unk_token_id

    train = json.loads(Path(f"data/2-3_regulated_verse/couplets_{args.meter}_train.json").read_text(encoding="utf-8"))
    if args.limit:
        train = train[:args.limit]
    tests = {kind: json.loads(Path(f"data/2-3_regulated_verse/couplet_controls/{args.meter}_{kind}.json").read_text(encoding="utf-8"))
             for kind in ("inner", "outer")}
    train_trad = [to_traditional.convert(item["couplet"]) for item in train]
    test_trad = {kind: [to_traditional.convert(item["couplet"]) for item in items] for kind, items in tests.items()}
    length = LENGTHS[args.meter]
    train_lines = [line for c in train_trad for line in lines_of(c)]
    all_lines = train_lines + [line for cs in test_trad.values() for c in cs for line in lines_of(c)]
    if any(len(line) != length for line in all_lines):
        raise SystemExit(f"Lines not of {length} characters")

    rng = random.Random(f"cipher-{args.level}-{args.meter}-{args.seed}")
    if args.level == "char":
        mapping = char_cipher(train_lines, all_lines, ok, args.band, rng)
        encode_line = lambda line: "".join(mapping.get(c, c) for c in line)
        unmapped = Counter(c for line in all_lines for c in line if c not in mapping)
    else:
        mapping = word_cipher(train_lines, all_lines, args.meter, ok, rng)
        encode_line = lambda line: "".join(mapping[u] for u in word_units(line, args.meter))
        unmapped = Counter()
    encode = lambda couplet: "，".join(encode_line(line) for line in lines_of(couplet))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.meter}_{args.level}cipher"
    train_out = [{**item, "couplet": encode(trad), "source_couplet": trad, "condition": f"{args.level}cipher"}
                 for item, trad in zip(train, train_trad)]
    (out_dir / f"{stem}_train.json").write_text(json.dumps(train_out, ensure_ascii=False, indent=1) + "\n",
                                                encoding="utf-8")
    for kind, items in tests.items():
        test_out = [{"id": item["id"], "label": item["label"], "couplet": encode(trad), "source_couplet": trad}
                    for item, trad in zip(items, test_trad[kind])]
        (out_dir / f"{stem}_{kind}.json").write_text(json.dumps(test_out, ensure_ascii=False, indent=1) + "\n",
                                                     encoding="utf-8")
    (out_dir / f"{stem}_map.json").write_text(json.dumps({"level": args.level, "band": args.band, "seed": args.seed,
                                                          "mapping": mapping}, ensure_ascii=False, indent=1) + "\n",
                                              encoding="utf-8")
    labels = Counter(item["label"] for item in train_out)
    kept = sum(o["couplet"] == o["source_couplet"] for o in train_out)
    print(f"{stem}: {len(train_out)} training couplets, labels {dict(labels)}, {len(mapping)} units mapped, "
          f"{sum(unmapped.values())} character tokens left unmapped ({len(unmapped)} types), "
          f"{kept} couplets unchanged; written to {out_dir}")
    for o in train_out[:4]:
        print(f"  {o['label']} {o['source_couplet']}  ->  {o['couplet']}")


if __name__ == "__main__":
    main()
