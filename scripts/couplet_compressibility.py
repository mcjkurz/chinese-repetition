"""Are parallel (inner) couplets more compressible than outer couplets in GujiBERT?

Each held-out regulated-verse couplet is encoded on its own ("line1，line2").
At every layer, the character embeddings of its two lines (the comma left out)
are centred and decomposed by PCA. A couplet counts as more compressible when
its leading principal components take a larger share of the variance. Three
measures are recorded per couplet and layer:

- pc1: share of variance on the first component;
- top_k: cumulative share on the first k = line-length components (a perfectly
  parallel couplet, where the two characters in each position coincide, would
  need only k components);
- participation_ratio: (sum of eigenvalues)^2 / sum of squared eigenvalues, the
  effective number of dimensions (lower = more compressible).

These spectral measures ignore which character stands opposite which: shuffling
the second line leaves them unchanged. Three further measures use the pairing
of corresponding positions directly (a_i in the first line, b_i in the second):

- position_r2: share of the couplet's variance explained by position, i.e.
  1 - sum_i |a_i - b_i|^2 / 2 / total sum of squares; the variance kept when
  each pair of corresponding characters is replaced by its mean;
- position_r2_gain: position_r2 minus its mean when the second line is
  cyclically shifted against the first by 1 .. n-1 positions, so that only the
  correspondence of positions, not the overall similarity of the lines, counts;
- diagonal_cosine: mean cosine between corresponding characters minus mean
  cosine between non-corresponding characters of the two lines.

Inner couplets (positions 2-3, required to be parallel) are compared with outer
couplets (positions 1 and 4) with a Mann-Whitney U test, and within poems that
contribute both kinds with a Wilcoxon signed-rank test on per-poem means.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu, wilcoxon
from transformers import AutoModel, AutoTokenizer

from common import INNER_POSITIONS, OUTER_POSITIONS, select_device, split_couplet
from self_similarity_matrix import DEFAULT_MODEL

MEASURES = ("pc1", "top_k", "participation_ratio", "position_r2", "position_r2_gain", "diagonal_cosine")


def position_r2(first: np.ndarray, second: np.ndarray, total: float) -> float:
    return 1 - float(np.sum((first - second) ** 2)) / 2 / total


def spectrum_measures(embeddings: np.ndarray, k: int) -> dict[str, float]:
    """Measures for one couplet whose first k rows are the first line and last k rows the second."""
    centred = embeddings - embeddings.mean(axis=0, keepdims=True)
    eigenvalues = np.linalg.svd(centred, compute_uv=False) ** 2
    shares = eigenvalues / eigenvalues.sum()
    first, second = centred[:k], centred[k:]
    total = float(np.sum(centred ** 2))
    matched = position_r2(first, second, total)
    shifted = np.mean([position_r2(first, np.roll(second, shift, axis=0), total) for shift in range(1, k)])
    unit = centred / np.linalg.norm(centred, axis=1, keepdims=True)
    cosines = unit[:k] @ unit[k:].T
    off_diagonal = cosines[~np.eye(k, dtype=bool)]
    return {
        "pc1": float(shares[0]),
        "top_k": float(shares[:k].sum()),
        "participation_ratio": float(eigenvalues.sum() ** 2 / np.sum(eigenvalues ** 2)),
        "position_r2": matched,
        "position_r2_gain": float(matched - shifted),
        "diagonal_cosine": float(np.diag(cosines).mean() - off_diagonal.mean()),
    }


def couplet_measures(couplets: list[dict], batch_size: int, model, tokenizer, device) -> list[dict]:
    """Per-couplet measures at every layer; couplets whose lines do not tokenize one-to-one are skipped."""
    records = []
    for batch_start in range(0, len(couplets), batch_size):
        batch = couplets[batch_start:batch_start + batch_size]
        texts = [item["couplet"] for item in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            hidden = torch.stack(model(**encoded, output_hidden_states=True).hidden_states)
        hidden = hidden.permute(1, 0, 2, 3).float().cpu().numpy()  # (couplet, layer, token, dim)
        for item, states, ids in zip(batch, hidden, encoded["input_ids"].tolist()):
            first, second = split_couplet(item["couplet"])
            n = item["meter"]
            if first is None or len(first) != n or len(second) != n:
                continue
            # Expected token layout: [CLS] first-line chars , second-line chars [SEP].
            if ids[1 + 2 * n + 1] != tokenizer.sep_token_id:
                continue
            positions = list(range(1, 1 + n)) + list(range(2 + n, 2 + 2 * n))
            records.append({
                "poem_id": item["poem_id"],
                "position": item["position"],
                "meter": n,
                "layers": [spectrum_measures(states[layer, positions], n) for layer in range(states.shape[0])],
            })
    return records


def compare(records: list[dict], layer_count: int) -> list[dict]:
    inner = [r for r in records if r["position"] in INNER_POSITIONS]
    outer = [r for r in records if r["position"] in OUTER_POSITIONS]
    summary = []
    for layer in range(layer_count):
        entry = {"layer": layer, "inner_n": len(inner), "outer_n": len(outer)}
        for measure in MEASURES:
            inner_values = np.array([r["layers"][layer][measure] for r in inner])
            outer_values = np.array([r["layers"][layer][measure] for r in outer])
            # Paired by poem: mean over each poem's inner couplets vs mean over its outer couplets.
            by_poem: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"inner": [], "outer": []})
            for r in inner:
                by_poem[r["poem_id"]]["inner"].append(r["layers"][layer][measure])
            for r in outer:
                by_poem[r["poem_id"]]["outer"].append(r["layers"][layer][measure])
            pairs = np.array([(np.mean(v["inner"]), np.mean(v["outer"]))
                              for v in by_poem.values() if v["inner"] and v["outer"]])
            differences = pairs[:, 0] - pairs[:, 1]
            entry[measure] = {
                "inner_mean": float(inner_values.mean()),
                "outer_mean": float(outer_values.mean()),
                "inner_minus_outer": float(inner_values.mean() - outer_values.mean()),
                "mann_whitney_p": float(mannwhitneyu(inner_values, outer_values).pvalue),
                "paired_poems": int(len(pairs)),
                "paired_mean_difference": float(differences.mean()),
                "paired_share_inner_higher": float(np.mean(differences > 0)),
                "wilcoxon_p": float(wilcoxon(differences).pvalue),
            }
        summary.append(entry)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+",
                        default=["data/2-3_regulated_verse/couplets_penta_test.json", "data/2-3_regulated_verse/couplets_hepta_test.json"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="results/couplet_compressibility/summary.json")
    args = parser.parse_args()

    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    layer_count = int(model.config.num_hidden_layers) + 1

    payload = {"model": args.model, "measures": list(MEASURES), "datasets": {}}
    for path in args.data:
        couplets = json.loads(Path(path).read_text(encoding="utf-8"))
        records = couplet_measures(couplets, args.batch_size, model, tokenizer, device)
        summary = compare(records, layer_count)
        payload["datasets"][path] = {"input_count": len(couplets), "scored_count": len(records), "summary": summary}
        print(f"{path}: scored {len(records)} of {len(couplets)}")
        for entry in summary:
            print(f"  layer {entry['layer']:2d} " + "  ".join(
                f"{m} {entry[m]['inner_mean']:.4f}/{entry[m]['outer_mean']:.4f} (p={entry[m]['wilcoxon_p']:.1e})"
                for m in MEASURES))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
