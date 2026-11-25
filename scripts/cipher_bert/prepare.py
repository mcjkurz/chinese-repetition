"""Prepare an enciphered classical-Chinese corpus for MLM training of GujiBERT with a new lexicon.

1. Download a slice of Daizhige (garychowcmu/daizhigev20) from chosen collections, up to a byte budget per
   collection, into data/cipher_corpus/daizhige/raw/ (kept as downloaded; read as data only). The poetry collections and the
   prose anthologies are left out, since they may contain the test couplets, the Qinding Sishu wen essays and
   the Guwen guanzhi pieces; the verse comes from the project's own training couplets instead
   (data/2-3_regulated_verse/couplets_{penta,hepta}_train.json, split by poem from the test couplets).
2. Convert everything to traditional characters with OpenCC (s2t), as the project does.
3. Drop every clause (text between punctuation, at least 4 characters) that also occurs in a test set: the
   held-out inner and outer couplets (data/2-3_regulated_verse/couplet_controls/{meter}_{inner,outer}.json), the 100 regular essays
   (data/4_eight_legged_essays/qinding_regular100/legs.json) and the Guwen guanzhi pseudo-essays (data/4_eight_legged_essays/qinding_regular100/guwen_pseudo.json).
4. Build one cipher over GujiBERT's vocabulary: CJK characters that GujiBERT reads as single known tokens, in the
   corpus or in the test sets, are ranked by frequency (corpus and test sets together) and permuted at random within
   bands of --band consecutive ranks, without fixed points. Punctuation and all other characters stay as they are.
5. Write, under --out: real_ids.npy and cipher_ids.npy (the corpus as GujiBERT token ids, whitespace removed),
   cipher.json (character -> character), manifest.json (files, sizes, counts).
"""

import argparse
import json
import random
import re
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import opencc
from transformers import AutoTokenizer

MODEL = "hsc748NLP/GujiBERT_fan"
REPO_RAW = "https://raw.githubusercontent.com/garychowcmu/daizhigev20/master/"
TREE = "https://api.github.com/repos/garychowcmu/daizhigev20/git/trees/master?recursive=1"
# Collection -> byte budget (UTF-8; about 3 bytes per character).
BUDGET = {"史藏/正史": 40e6, "子藏/笔记": 25e6, "儒藏/语录": 20e6, "史藏/诏令奏议": 15e6, "史藏/编年": 15e6,
          "儒藏/四书": 10e6, "诗藏/诗话": 10e6}
EXCLUDE_NAMES = ("四书文", "四書文", "古文观止", "古文觀止")
MAX_FILE = 15e6
CLAUSE = re.compile(r"[，。；：！？、,.;:!?「」『』（）()《》〈〉“”‘’\s]+")
ROOT = Path(__file__).resolve().parents[2]


def is_cjk(ch: str) -> bool:
    return unicodedata.category(ch) == "Lo"


def select_files(tree: list, rng: random.Random) -> list:
    chosen = []
    for prefix, budget in BUDGET.items():
        files = [x for x in tree if x["type"] == "blob" and x["path"].startswith(prefix + "/")
                 and x["path"].endswith(".txt") and x["size"] <= MAX_FILE
                 and not any(n in x["path"] for n in EXCLUDE_NAMES)]
        rng.shuffle(files)
        total = 0
        for x in files:
            if total >= budget:
                break
            chosen.append(x)
            total += x["size"]
    return chosen


def test_clauses(converter) -> tuple[set, str]:
    """Clauses of the test sets, and their full text (for the cipher's character inventory)."""
    texts = []
    for meter in ("penta", "hepta"):
        for kind in ("inner", "outer"):
            for item in json.loads((ROOT / f"data/2-3_regulated_verse/couplet_controls/{meter}_{kind}.json").read_text(encoding="utf-8")):
                texts += [item["line1_traditional"], item["line2_traditional"]]
    for name in ("legs.json", "guwen_pseudo.json"):
        data = json.loads((ROOT / "data/4_eight_legged_essays/qinding_regular100" / name).read_text(encoding="utf-8"))
        for essay in data["essays"]:
            for pair in essay["pairs"]:
                texts += [pair["a"], pair["b"]]
    full = "\n".join(texts)
    clauses = {c for t in texts for c in CLAUSE.split(t) if len(c) >= 4}
    clauses |= {converter.convert(c) for c in clauses}
    return clauses, full


def filter_clauses(text: str, banned: set) -> tuple[str, int]:
    """Remove banned clauses (with the punctuation after them) from the text."""
    parts = re.split(f"({CLAUSE.pattern})", text)
    kept, dropped = [], 0
    for i in range(0, len(parts), 2):
        clause = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        if len(clause) >= 4 and clause in banned:
            dropped += 1
            continue
        kept.append(clause + sep)
    return "".join(kept), dropped


def build_cipher(counts: dict, band: int, rng: random.Random) -> dict:
    ranked = sorted(counts, key=lambda c: (-counts[c], c))
    mapping = {}
    for start in range(0, len(ranked), band):
        group = ranked[start:start + band]
        if len(group) < 2:
            mapping.update({c: c for c in group})
            continue
        while True:
            perm = group[:]
            rng.shuffle(perm)
            if all(a != b for a, b in zip(group, perm)):
                break
        mapping.update(dict(zip(group, perm)))
    return mapping


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default=str(ROOT / "data/cipher_corpus/daizhige/raw"))
    p.add_argument("--out", default=str(ROOT / "results/cipher_bert/corpus"))
    p.add_argument("--band", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    raw_dir, out = Path(args.raw_dir), Path(args.out)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    tree = json.loads(urllib.request.urlopen(TREE, timeout=60).read())["tree"]
    files = select_files(tree, rng)
    print(f"{len(files)} files, {sum(x['size'] for x in files) / 1e6:.1f} MB", flush=True)
    for x in files:
        target = raw_dir / x["path"]
        if target.exists() and target.stat().st_size == x["size"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = REPO_RAW + urllib.parse.quote(x["path"])
        target.write_bytes(urllib.request.urlopen(url, timeout=120).read())

    converter = opencc.OpenCC("s2t")
    banned, test_text = test_clauses(converter)
    documents, dropped = [], 0
    for x in files:
        text = converter.convert((raw_dir / x["path"]).read_text(encoding="utf-8", errors="replace"))
        text, n = filter_clauses(text, banned)
        documents.append(text)
        dropped += n
    for meter in ("penta", "hepta"):
        couplets = json.loads((ROOT / f"data/2-3_regulated_verse/couplets_{meter}_train.json").read_text(encoding="utf-8"))
        text = "".join(converter.convert(c["couplet"]) + "。" for c in couplets)
        text, n = filter_clauses(text, banned)
        documents.append(text)
        dropped += n
    corpus = "\n".join(documents)

    tok = AutoTokenizer.from_pretrained(MODEL)
    vocab = tok.get_vocab()
    counts = {}
    for ch in corpus + test_text:
        if is_cjk(ch) and ch in vocab:
            counts[ch] = counts.get(ch, 0) + 1
    cipher = build_cipher(counts, args.band, rng)
    unk = tok.unk_token_id
    real_ids = np.array([vocab.get(ch, unk) for ch in corpus if not ch.isspace()], dtype=np.int32)
    lookup = np.arange(len(vocab), dtype=np.int32)
    for a, b in cipher.items():
        lookup[vocab[a]] = vocab[b]
    cipher_ids = lookup[real_ids]
    np.save(out / "real_ids.npy", real_ids)
    np.save(out / "cipher_ids.npy", cipher_ids)
    (out / "cipher.json").write_text(json.dumps(cipher, ensure_ascii=False), encoding="utf-8")
    manifest = {"files": [{"path": x["path"], "bytes": x["size"]} for x in files], "budget": BUDGET,
                "tokens": int(len(real_ids)), "unk_share": float((real_ids == unk).mean()),
                "cipher_size": len(cipher), "fixed_points": sum(a == b for a, b in cipher.items()),
                "test_clauses": len(banned), "clauses_dropped": dropped, "band": args.band, "seed": args.seed}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("files", "budget")}, indent=1))


if __name__ == "__main__":
    main()
