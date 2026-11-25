"""Transition alignment of inner and outer couplets: GujiBERT on real text vs the cipher model on enciphered text.

For each meter, the 1,000 held-out inner and 1,000 outer couplets (data/2-3_regulated_verse/couplet_controls/{meter}_{inner,outer}.json)
are encoded as "line1，line2。" (transition_analysis.py) by
- GujiBERT (hsc748NLP/GujiBERT_fan) on the real characters, and
- the cipher model (GujiBERT with the word embeddings of train.py) on the characters enciphered with cipher.json.
At every layer, a couplet's alignment is the mean cosine between corresponding character-to-character transitions
of its two lines. Each model gets its own position-matched random-pair baseline from the same 2,000 couplets:
the transition at slot k of a first line against the transition at slot k of the second line of another couplet
(--baseline-pairs random partners per couplet).

Writes results/cipher_bert/alignment/{meter}.json (per-couplet values and baselines) and summary.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from transition_analysis import (DEFAULT_MODEL, couplet_context_and_targets,  # noqa: E402
                                 get_character_embeddings_batch, token_count_matches_text_length)

METERS = {"penta": "Pentasyllabic", "hepta": "Heptasyllabic"}


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def encode(lines: list[tuple[str, str]], tokenizer, model, device, batch_size: int):
    """Unit transition vectors (n, layers, slots, hidden) of both lines, and the indices kept."""
    t1, t2, kept = [], [], []
    for start in range(0, len(lines), batch_size):
        batch = []
        for idx in range(start, min(start + batch_size, len(lines))):
            text, targets = couplet_context_and_targets(*lines[idx], True)
            if token_count_matches_text_length(text, tokenizer):
                batch.append((idx, text, targets))
        if not batch:
            continue
        embeddings = get_character_embeddings_batch([b[1] for b in batch], [b[2] for b in batch],
                                                    tokenizer, model, device)
        for (idx, _, _), layers in zip(batch, embeddings):
            n = len(lines[idx][0])
            t1.append(unit(np.diff(layers[:, :n], axis=1)))
            t2.append(unit(np.diff(layers[:, n:], axis=1)))
            kept.append(idx)
    return np.array(t1), np.array(t2), kept


def alignment_and_baseline(t1, t2, rng, pairs: int):
    aligned = np.einsum("nlsh,nlsh->nls", t1, t2).mean(axis=2)  # (n, layers)
    n = len(t1)
    sims = []
    for _ in range(pairs):
        partner = (np.arange(n) + rng.integers(1, n, size=n)) % n  # never the couplet itself
        sims.append(np.einsum("nlsh,nlsh->nls", t1, t2[partner]).mean(axis=2).mean(axis=0))
    return aligned, np.mean(sims, axis=0)


def cohens_d(x, y):
    pooled = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    return float((x.mean() - y.mean()) / pooled)


def auc(x, y):
    """Probability that a random x exceeds a random y."""
    ranks = np.argsort(np.argsort(np.r_[x, y])) + 1
    return float((ranks[: len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings", default=str(ROOT / "results/cipher_bert/frozen_body/embeddings_final.pt"))
    p.add_argument("--cipher", default=str(ROOT / "results/cipher_bert/corpus/cipher.json"))
    p.add_argument("--out", default=str(ROOT / "results/cipher_bert/alignment"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--baseline-pairs", type=int, default=20)
    p.add_argument("--layer", type=int, default=8)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cipher = json.loads(Path(args.cipher).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

    real_model = AutoModel.from_pretrained(DEFAULT_MODEL, output_hidden_states=True).to(device).eval()
    cipher_model = AutoModel.from_pretrained(DEFAULT_MODEL, output_hidden_states=True)
    trained = torch.load(args.embeddings, map_location="cpu")
    cipher_model.embeddings.word_embeddings.weight.data.copy_(trained["bert.embeddings.word_embeddings.weight"])
    cipher_model.to(device).eval()

    rows = []
    for meter, meter_name in METERS.items():
        sets = {kind: json.loads((ROOT / f"data/2-3_regulated_verse/couplet_controls/{meter}_{kind}.json").read_text(encoding="utf-8"))
                for kind in ("inner", "outer")}
        lines = [(c["line1_traditional"], c["line2_traditional"]) for kind in ("inner", "outer") for c in sets[kind]]
        labels = np.array([kind for kind in ("inner", "outer") for _ in sets[kind]])
        enciphered = [("".join(cipher.get(ch, ch) for ch in a), "".join(cipher.get(ch, ch) for ch in b)) for a, b in lines]
        result = {"meter": meter}
        for name, model, data in (("GujiBERT, real text", real_model, lines), ("Cipher model, enciphered text", cipher_model, enciphered)):
            t1, t2, kept = encode(data, tokenizer, model, device, args.batch_size)
            aligned, baseline = alignment_and_baseline(t1, t2, np.random.default_rng(0), args.baseline_pairs)
            corrected = aligned - baseline
            kind = labels[kept]
            inner, outer = corrected[kind == "inner"], corrected[kind == "outer"]
            result[name] = {"kept": len(kept), "baseline": baseline.tolist(),
                            "labels": kind.tolist(), "corrected": np.round(corrected, 4).tolist(),
                            "inner_mean": inner.mean(axis=0).tolist(), "outer_mean": outer.mean(axis=0).tolist(),
                            "d": [cohens_d(inner[:, l], outer[:, l]) for l in range(inner.shape[1])],
                            "auc": [auc(inner[:, l], outer[:, l]) for l in range(inner.shape[1])]}
            L = args.layer
            rows.append(f"| {meter_name} | {name} | {len(kept)} | {inner[:, L].mean():.3f} | {outer[:, L].mean():.3f} | "
                        f"{cohens_d(inner[:, L], outer[:, L]):.2f} | {auc(inner[:, L], outer[:, L]):.2f} | "
                        f"{min(result[name]['d']):.2f}–{max(result[name]['d']):.2f} |")
            print(rows[-1], flush=True)
        (out / f"{meter}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    header = [f"Transition alignment minus random-pair baseline at layer {args.layer}; d and AUC: inner vs outer.", "",
              "| Meter | Model | Couplets | Inner | Outer | d | AUC | d, all layers |", "|:--|:--|--:|--:|--:|--:|--:|:--:|"]
    (out / "summary.md").write_text("\n".join(header + rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
