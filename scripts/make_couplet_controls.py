"""Control couplets for the decoding and encoding experiments on regulated verse.

For each meter, 1,000 inner and 1,000 outer couplets are sampled from the
held-out test set (data/2-3_regulated_verse/couplets_{meter}_test.json), 200 per dynasty, among
couplets whose characters all belong to the --vocabulary most frequent
characters of the couplet corpus (training and test sets of both meters).
Control sets are built on the sampled inner couplets, couplet by couplet:

- nonsense: every distinct character of a couplet is replaced by a random
  character of the same word class from the vocabulary, the same replacement
  wherever the character recurs in the couplet, and never a character of the
  couplet itself, so that the grammatical parallelism of the two lines is kept
  and their sense is lost. Word classes are assigned by hand, one per
  character, for its dominant use in regulated verse (data/2-3_regulated_verse/couplet_controls/
  lexicon/classes_*.tsv; see docs/couplet_controls_brief.md);
- pair-swapped: every pair of corresponding characters (position k of the
  first and of the second line) is taken from position k of another inner
  couplet of the set, a different one for every position, so that every
  vertical pair is a real parallel pair while neither line makes sense;
- segment-swapped: the same with corresponding segments of the line's rhythm
  (2+3 characters in pentasyllabic, 2+2+3 in heptasyllabic lines), so that
  two-character words are kept whole;
- word-swapped: the same with the word units of the line (2+2+1 characters in
  pentasyllabic, 2+2+2+1 in heptasyllabic lines, reading the final
  three-character segment as 2+1), so that words are kept whole but no two
  words of a line come from the same couplet;
- char-shuffled: the characters of the second line are permuted (a
  derangement wherever the line's characters allow one, and never the line
  itself), so that both lines keep their characters while the second no longer
  answers the first position by position;
- composed: parallel couplets written by hand after each inner couplet
  (data/2-3_regulated_verse/couplet_controls/composed/composed_*.txt), on another subject, with
  the same number of characters per line, no run of three characters taken
  over from the source couplet, no line found verbatim in the corpus, and
  every character in the vocabulary.

All sets are written in the format of data/2-3_regulated_verse/couplets_{meter}_alignment.json
(simplified and traditional fields), so that decoding_io.py
(--test-data-path-template) and transition_analysis.py (--data-path-template)
read them unchanged.

    sample            sample the real couplets and write the lists to classify
    check-lexicon     check the word-class files
    nonsense          build the nonsense couplets
    pairswap          build the pair-swapped couplets
    segswap           build the segment-swapped couplets
    wordswap          build the word-swapped couplets
    charshuf          build the char-shuffled couplets
    check-composed    check the composed couplets (all files, or those given)
    composed          build the composed set (only without problems)
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

import opencc
from transformers import AutoTokenizer

METERS = {"penta": 5, "hepta": 7}
DYNASTIES = ["唐", "宋", "元", "明", "清"]
CLASSES = {"N": "noun", "V": "verb", "A": "adjective or stative verb", "C": "colour", "M": "numeral or quantity",
           "D": "direction or position", "P": "pronoun", "F": "function word: adverb, preposition, conjunction, particle"}
OUT = Path("data/2-3_regulated_verse/couplet_controls")
PUNCTUATION = re.compile(r"[，。]")
MODELS = ["hsc748NLP/GujiBERT_fan", "hsc748NLP/GujiGPT_fan"]
to_traditional = opencc.OpenCC("s2t")


def corpus_lines() -> list[str]:
    lines = []
    for meter in METERS:
        for split in ("train", "test"):
            for item in json.loads(Path(f"data/2-3_regulated_verse/couplets_{meter}_{split}.json").read_text(encoding="utf-8")):
                lines.extend(PUNCTUATION.split(item["couplet"])[:2])
    return lines


def record(line1: str, line2: str, meter: str, **fields) -> dict:
    """A couplet in the format of the alignment data."""
    t1, t2 = to_traditional.convert(line1), to_traditional.convert(line2)
    return {"couplet": f"{line1}，{line2}", "couplet_traditional": f"{t1}，{t2}", "line1": line1, "line2": line2,
            "line1_traditional": t1, "line2_traditional": t2, "meter": METERS[meter], **fields}


def single_tokens(text: str, tokenizers) -> bool:
    return all(len(t.tokenize(c)) == 1 and t.convert_tokens_to_ids(c) != t.unk_token_id
               for t in tokenizers for c in to_traditional.convert(text))


def sample(args) -> None:
    lines = corpus_lines()
    freq = collections.Counter("".join(lines))
    vocabulary = [c for c, _ in freq.most_common(args.vocabulary)]
    allowed = set(vocabulary)
    rng = random.Random(args.seed)
    for meter in METERS:
        test = json.loads(Path(f"data/2-3_regulated_verse/couplets_{meter}_test.json").read_text(encoding="utf-8"))
        for label, name in (("i", "inner"), ("o", "outer")):
            chosen = []
            for dynasty in DYNASTIES:
                pool = [x for x in test if x["label"] == label and x["dynasty"] == dynasty
                        and set(PUNCTUATION.sub("", x["couplet"])) <= allowed]
                chosen += rng.sample(pool, args.per_dynasty)
            out = [record(*PUNCTUATION.split(x["couplet"])[:2], meter, id=f"{meter}-{name}-{i:04d}",
                          position=x["position"], label=label, dynasty=x["dynasty"], poem_id=x["poem_id"])
                   for i, x in enumerate(chosen)]
            write(OUT / f"{meter}_{name}.json", out)
    # Characters to classify, with two corpus lines each as examples of use.
    examples = collections.defaultdict(list)
    for line in lines:
        for c in set(line):
            if c in allowed and len(examples[c]) < 2:
                examples[c].append(line)
    lexicon = OUT / "lexicon"
    lexicon.mkdir(parents=True, exist_ok=True)
    size = -(-len(vocabulary) // args.chunks)
    for k in range(args.chunks):
        chunk = vocabulary[k * size:(k + 1) * size]
        (lexicon / f"chars_{k + 1}.txt").write_text(
            "".join(f"{c}\t{freq[c]}\t{' / '.join(examples[c])}\n" for c in chunk), encoding="utf-8")
    print(f"{len(vocabulary)} characters in {args.chunks} lists: {lexicon}")


def load_lexicon() -> tuple[dict[str, str], list[str]]:
    classes, problems = {}, []
    for path in sorted((OUT / "lexicon").glob("classes_*.tsv")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.startswith("#"):
                continue
            char, _, tag = line.partition("\t")
            tag = tag.strip().split("\t")[0]
            if len(char) != 1 or tag not in CLASSES:
                problems.append(f"{path.name}:{number}: cannot read {line[:20]!r}")
            elif char in classes and classes[char] != tag:
                problems.append(f"{path.name}:{number}: {char} classified twice ({classes[char]}, {tag})")
            else:
                classes[char] = tag
    return classes, problems


def check_lexicon(args) -> dict[str, str]:
    classes, problems = load_lexicon()
    for path in sorted((OUT / "lexicon").glob("chars_*.txt")):
        chars = [line.split("\t")[0] for line in path.read_text(encoding="utf-8").splitlines() if line]
        missing = [c for c in chars if c not in classes]
        if missing:
            problems.append(f"{path.name}: {len(missing)} characters not classified ({''.join(missing[:20])} ...)")
    print("\n".join(problems))
    print(f"{len(classes)} characters classified: " +
          ", ".join(f"{k} {v}" for k, v in sorted(collections.Counter(classes.values()).items())) +
          f"; {len(problems)} problems")
    return classes


def nonsense(args) -> None:
    classes, problems = load_lexicon()
    if problems:
        raise SystemExit("\n".join(problems))
    tokenizers = [AutoTokenizer.from_pretrained(m) for m in MODELS]
    pools = collections.defaultdict(list)
    for char, tag in classes.items():
        if single_tokens(char, tokenizers):
            pools[tag].append(char)
    for meter in METERS:
        source = json.loads((OUT / f"{meter}_inner.json").read_text(encoding="utf-8"))
        out = []
        for index, item in enumerate(source):
            rng = random.Random(args.seed + index)
            text = item["line1"] + item["line2"]
            mapping = {}
            for char in dict.fromkeys(text):
                if char not in classes:
                    raise SystemExit(f"{item['id']}: {char} has no word class")
                used = set(text) | set(mapping.values())
                choices = [c for c in pools[classes[char]] if c not in used]
                mapping[char] = rng.choice(choices)
            line1, line2 = ("".join(mapping[c] for c in line) for line in (item["line1"], item["line2"]))
            out.append(record(line1, line2, meter, id=item["id"].replace("inner", "nonsense"), template=item["id"],
                              position=item["position"], label="i", dynasty=item["dynasty"], poem_id=item["id"],
                              classes="".join(classes[c] for c in item["line1"]) + "/" +
                                      "".join(classes[c] for c in item["line2"])))
        write(OUT / f"{meter}_nonsense.json", out)


SEGMENTS = {"penta": [(0, 2), (2, 5)], "hepta": [(0, 2), (2, 4), (4, 7)]}
# Word units of the line, with the three-character segment read as 2+1 (the common reading).
WORDS = {"penta": [(0, 2), (2, 4), (4, 5)], "hepta": [(0, 2), (2, 4), (4, 6), (6, 7)]}


def pairswap(args, segments: bool = False, words: bool = False) -> None:
    name = "segswap" if segments else "wordswap" if words else "pairswap"
    for meter in METERS:
        source = json.loads((OUT / f"{meter}_inner.json").read_text(encoding="utf-8"))
        out = []
        for index, item in enumerate(source):
            rng = random.Random(args.seed + index)
            spans = (SEGMENTS[meter] if segments else WORDS[meter] if words
                     else [(k, k + 1) for k in range(len(item["line1"]))])
            donors = rng.sample([x for x in source if x is not item], len(spans))
            line1 = "".join(d["line1"][a:b] for (a, b), d in zip(spans, donors))
            line2 = "".join(d["line2"][a:b] for (a, b), d in zip(spans, donors))
            out.append(record(line1, line2, meter, id=item["id"].replace("inner", name), template=item["id"],
                              position=item["position"], label="i", dynasty=item["dynasty"], poem_id=item["id"],
                              donors=[d["id"] for d in donors]))
        write(OUT / f"{meter}_{name}.json", out)


def shuffle_line(line: str, rng: random.Random, tries: int = 200) -> str:
    """A permutation of the line with as few characters left in place as possible, never the line itself."""
    best = None
    for _ in range(tries):
        order = list(range(len(line)))
        rng.shuffle(order)
        new = "".join(line[i] for i in order)
        fixed = sum(a == b for a, b in zip(new, line))
        if new != line and (best is None or fixed < best[0]):
            best = (fixed, new)
            if fixed == 0:
                break
    if best is None:
        raise ValueError(f"{line}: no permutation differs from the line")
    return best[1]


def charshuf(args) -> None:
    for meter in METERS:
        source = json.loads((OUT / f"{meter}_inner.json").read_text(encoding="utf-8"))
        out = []
        for index, item in enumerate(source):
            rng = random.Random(args.seed + index)
            out.append(record(item["line1"], shuffle_line(item["line2"], rng), meter,
                              id=item["id"].replace("inner", "charshuf"), template=item["id"],
                              position=item["position"], label="i", dynasty=item["dynasty"], poem_id=item["id"]))
        write(OUT / f"{meter}_charshuf.json", out)


def read_composed(paths: list[Path]) -> dict[str, tuple[str, str, Path, int]]:
    composed = {}
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.startswith("#"):
                continue
            key, _, couplet = line.partition("\t")
            parts = PUNCTUATION.split(couplet.strip().rstrip("。"))
            composed[key.strip()] = (parts[0], parts[1] if len(parts) > 1 else "", path, number)
    return composed


def check_composed(args, quiet: bool = False) -> tuple[list[str], dict]:
    paths = [Path(p) for p in args.files] or sorted((OUT / "composed").glob("composed_*.txt"))
    composed = read_composed(paths)
    sources = {x["id"]: x for meter in METERS
               for x in json.loads((OUT / f"{meter}_inner.json").read_text(encoding="utf-8"))}
    corpus = set(corpus_lines())
    freq = collections.Counter("".join(corpus))
    allowed = {c for c, _ in freq.most_common(args.vocabulary)}
    tokenizers = [AutoTokenizer.from_pretrained(m) for m in MODELS]
    problems = []
    for key, (line1, line2, path, number) in composed.items():
        where = f"{path.name}:{number} ({key})"
        source = sources.get(key)
        if source is None:
            problems.append(f"{where}: no source couplet with this id")
            continue
        n = source["meter"]
        if len(line1) != n or len(line2) != n:
            problems.append(f"{where}: lines of {len(line1)} and {len(line2)} characters instead of {n}")
            continue
        text = line1 + line2
        if not set(text) <= allowed:
            problems.append(f"{where}: outside the vocabulary {''.join(sorted(set(text) - allowed))}")
        if not single_tokens(text, tokenizers):
            problems.append(f"{where}: a character is not one model token")
        for line in (line1, line2):
            if line in corpus:
                problems.append(f"{where}: {line} occurs verbatim in the corpus")
        real = source["line1"] + "，" + source["line2"]
        copied = {line[i:i + 3] for line in (line1, line2) for i in range(len(line) - 2) if line[i:i + 3] in real}
        if copied:
            problems.append(f"{where}: copied from the source {' '.join(sorted(copied))}")
        if line1 == line2:
            problems.append(f"{where}: the two lines are identical")
    if not quiet:
        print("\n".join(problems))
        print(f"{len(composed)} composed couplets checked, {len(problems)} problems")
    return problems, composed


def build_composed(args) -> None:
    args.files = []
    problems, composed = check_composed(args, quiet=True)
    if problems:
        raise SystemExit("\n".join(problems) + f"\n{len(problems)} problems; nothing written")
    for meter in METERS:
        source = json.loads((OUT / f"{meter}_inner.json").read_text(encoding="utf-8"))
        missing = [x["id"] for x in source if x["id"] not in composed]
        if missing:
            raise SystemExit(f"{meter}: {len(missing)} couplets not composed yet ({' '.join(missing[:5])} ...)")
        write(OUT / f"{meter}_composed.json",
              [record(composed[x["id"]][0], composed[x["id"]][1], meter, id=x["id"].replace("inner", "composed"),
                      template=x["id"], position=x["position"], label="i", dynasty=x["dynasty"], poem_id=x["id"])
               for x in source])


def write(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(items)} couplets: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sample", "check-lexicon", "nonsense", "pairswap", "segswap", "wordswap",
                                            "charshuf", "check-composed", "composed"])
    parser.add_argument("files", nargs="*", help="With check-composed: files to check (default: all)")
    parser.add_argument("--vocabulary", type=int, default=3000)
    parser.add_argument("--per-dynasty", type=int, default=200)
    parser.add_argument("--chunks", type=int, default=6, help="Lists of characters to classify")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"sample": sample, "check-lexicon": check_lexicon, "nonsense": nonsense, "pairswap": pairswap,
     "segswap": lambda a: pairswap(a, segments=True),
     "wordswap": lambda a: pairswap(a, words=True), "charshuf": charshuf,
     "check-composed": check_composed, "composed": build_composed}[args.command](args)


if __name__ == "__main__":
    main()
