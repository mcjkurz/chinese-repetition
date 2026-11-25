"""Leg alignment of the 100 regular eight-legged essays: GujiBERT on real text vs the cipher model on enciphered text.

The measure is that of the essay figure (scripts/baguwen_pseudo_legs.py, whose functions are reused): each leg is
split into clauses at its punctuation, each clause is encoded on its own and represented at every layer by the mean
of its characters' embeddings, and the alignment of a pair is the mean cosine between corresponding clause-to-clause
transitions of its two legs, minus a random-pair baseline from --random-pairs pseudo-leg pairs of non-parallel prose
with the same number of clauses. Conditions: the regular essays (data/4_eight_legged_essays/qinding_regular100/legs.json) and the matched
Guwen guanzhi pseudo-essays (data/4_eight_legged_essays/qinding_regular100/guwen_pseudo.json); baseline pool: pseudo-leg windows of
data/4_eight_legged_essays/pseudo_legs/guwen.json, as in the article. The same pairs and the same random partners are used for both
models; for the cipher model every clause is enciphered with cipher.json (punctuation, and so the clause division,
is unchanged).

Writes results/cipher_bert/essays/results.json and summary.md.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from baguwen_pseudo_legs import alignment, cohens_d, load_corpus, pair_steps, prose_windows, text_means  # noqa: E402

MODEL = "hsc748NLP/GujiBERT_fan"
CONDITIONS = {"Regular essays": "data/4_eight_legged_essays/qinding_regular100/legs.json",
              "Guwen guanzhi pseudo-essays": "data/4_eight_legged_essays/qinding_regular100/guwen_pseudo.json"}
POOL = "data/4_eight_legged_essays/pseudo_legs/guwen.json"


def auc(x, y):
    ranks = np.argsort(np.argsort(np.r_[x, y])) + 1
    return float((ranks[: len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings", default=str(ROOT / "results/cipher_bert/frozen_body/embeddings_final.pt"))
    p.add_argument("--cipher", default=str(ROOT / "results/cipher_bert/corpus/cipher.json"))
    p.add_argument("--out", default=str(ROOT / "results/cipher_bert/essays"))
    p.add_argument("--min-fragments", type=int, default=3)
    p.add_argument("--pairs-per-n", type=int, default=300)
    p.add_argument("--random-pairs", type=int, default=30)
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cipher = json.loads(Path(args.cipher).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    encode_args = SimpleNamespace(pairs_per_batch=16, batch_size=256)
    rng = random.Random(args.seed)

    # The same pairs for both models: essays and pseudo-essays, and the baseline pool of prose windows.
    pairs = {label: load_corpus(str(ROOT / path), args.min_fragments) for label, path in CONDITIONS.items()}
    fragments = [f for a, b, _, _ in pairs["Regular essays"] for f in a + b]
    min_chars, max_chars = min(map(len, fragments)), max(map(len, fragments))
    sizes = sorted({len(a) for a, _, _, _ in pairs["Regular essays"]})
    pieces = json.loads((ROOT / POOL).read_text(encoding="utf-8"))["pieces"]
    def clean(fragment: str) -> bool:  # one token per character, in real and in enciphered characters
        enciphered = "".join(cipher.get(ch, ch) for ch in fragment)
        return all(len(tokenizer.tokenize(f)) == len(f) and tokenizer.unk_token not in tokenizer.tokenize(f)
                   for f in (fragment, enciphered))

    pool_pairs = []
    for n in sizes:
        windows = [w for w in prose_windows(pieces, n, min_chars, max_chars) if all(map(clean, w[0] + w[1]))]
        pool_pairs += rng.sample(windows, min(args.pairs_per_n, len(windows)))
    # Random partners, fixed in advance by index so that both models use the same ones.
    pool_by_n = {n: [i for i, w in enumerate(pool_pairs) if len(w[0]) == n] for n in sizes}
    partners = {label: [rng.sample(pool_by_n[len(a)], min(args.random_pairs, len(pool_by_n[len(a)])))
                        for a, _, _, _ in records] for label, records in pairs.items()}

    def encipher(pair_list):
        return [(["".join(cipher.get(ch, ch) for ch in f) for f in a], ["".join(cipher.get(ch, ch) for ch in f) for f in b], t, pos)
                for a, b, t, pos in pair_list]

    real_model = AutoModel.from_pretrained(MODEL).to(device).eval()
    cipher_model = AutoModel.from_pretrained(MODEL)
    trained = torch.load(args.embeddings, map_location="cpu")
    cipher_model.embeddings.word_embeddings.weight.data.copy_(trained["bert.embeddings.word_embeddings.weight"])
    cipher_model.to(device).eval()

    results, rows = {}, []
    L = args.layer
    for model_label, model, transform in (("GujiBERT, real text", real_model, lambda x: x),
                                          ("Cipher model, enciphered text", cipher_model, encipher)):
        pool = pair_steps(transform(pool_pairs), encode_args, model, tokenizer, device)
        if len(pool) != len(pool_pairs):
            raise SystemExit(f"{model_label}: {len(pool_pairs) - len(pool)} baseline pairs did not tokenize")
        entry = {}
        for label, records in pairs.items():
            encoded = pair_steps(transform(records), encode_args, model, tokenizer, device)
            if len(encoded) != len(records):
                raise SystemExit(f"{model_label}, {label}: {len(records) - len(encoded)} pairs did not tokenize")
            for record, chosen in zip(encoded, partners[label]):
                record["alignment"] = alignment(record["A"], record["B"])
                record["baseline"] = np.mean([alignment(record["A"], pool[i]["B"]) for i in chosen], axis=0)
            diff = np.array([r["alignment"] - r["baseline"] for r in encoded])
            entry[label] = {"pairs": len(encoded), "difference": diff.mean(axis=0).tolist(),
                            "text_means": text_means(encoded, L)}
            if model_label.startswith("Cipher") and label == "Regular essays":
                # Per-pair records in the format of results/baguwen_regular100/pair_alignment/pairs.json, so that
                # the essay figure can show the enciphered essays as its nonsense condition.
                records_out = [{"text": r["text"], "position": r["position"], "a": r["a"], "b": r["b"],
                                "alignment": np.round(r["alignment"], 6).tolist(),
                                "baseline": np.round(r["baseline"], 6).tolist()} for r in encoded]
                (out / "pairs.json").write_text(json.dumps({"Nonsense": records_out}, ensure_ascii=False),
                                                encoding="utf-8")
        essays = np.array(list(entry["Regular essays"]["text_means"].values()))
        prose = np.array(list(entry["Guwen guanzhi pseudo-essays"]["text_means"].values()))
        entry["layer"] = L
        entry["essays_vs_prose"] = {"d": cohens_d(essays, prose), "auc": auc(essays, prose)}
        results[model_label] = entry
        rows.append(f"| {model_label} | {essays.mean():.3f} | {prose.mean():.3f} | {cohens_d(essays, prose):.2f} | "
                    f"{auc(essays, prose):.2f} |")
        print(rows[-1], flush=True)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    header = [f"Leg alignment minus random-pair baseline at layer {L}, mean of each text's pairs (100 texts per set); "
              "d and AUC: regular essays vs Guwen guanzhi pseudo-essays.", "",
              "| Model | Regular essays | Guwen guanzhi | d | AUC |", "|:--|--:|--:|--:|--:|"]
    (out / "summary.md").write_text("\n".join(header + rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
