"""Leg-level alignment of regular eight-legged essays: the odd legs against the even legs.

Measure 2 of docs/baguwen_leg_alignment_method.md. Every essay of a --corpus
or --controls file with exactly four leg pairs is read as two streams of four
legs, the odd legs (1A, 2A, 3A, 4A: legs 1, 3, 5, 7) and the even legs (1B,
2B, 3B, 4B: legs 2, 4, 6, 8). Every fragment is encoded on its own by GujiBERT
as the mean of its characters' embeddings, as in baguwen_pseudo_legs.py, and
a leg is the mean of its fragment vectors, so that one-fragment legs count and
partner legs need not have the same number of fragments. The alignment of an
essay is the mean cosine between corresponding unit transitions of its two
streams (leg 1→3 against 2→4, 3→5 against 4→6, 5→7 against 6→8), at every
layer.

The prose comparison (--prose) is one pseudo-essay of 古文觀止 per real essay
(make_pseudo_essays.py): consecutive fragments of one piece cut into eight
pseudo-legs with the fragment counts of the real essay's legs. Two baselines,
for every essay: the odd stream against the even streams of --random-pairs
prose pseudo-essays (the main baseline, as for measure 1), and against the
even streams of --random-pairs other essays of the same condition (which
keeps the genre but not the topic, so that the difference shows how much of
the raw alignment the shared topic of one essay accounts for).

The essay is the unit: per condition, the mean difference from the baseline
with a bootstrap 95% confidence interval, Mann-Whitney test and Cohen's d
against the prose pseudo-essays, and a Wilcoxon signed-rank test of each real
essay against the control built on it. Writes summary.json, essays.json and a
figure.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from qhchina import load_fonts
from scipy.stats import mannwhitneyu, wilcoxon
from transformers import AutoModel, AutoTokenizer

from baguwen_fragment_space import FRAGMENT_SPLIT
from baguwen_pseudo_legs import HIGHLIGHT_COLOR, cohens_d, encode
from common import select_device
from self_similarity_matrix import DEFAULT_MODEL

COLORS = {"欽定四書文": "#2171b5", "Composed": "#6baed6", "Nonsense": "#9ecae1", "古文觀止": "#555559"}


def split(text: str) -> list[str]:
    return [f for f in FRAGMENT_SPLIT.split(text) if f]


def load_essays(path: str) -> list[dict]:
    """Essays with exactly four pairs, as their eight legs in text order (1A, 1B, 2A, ...)."""
    essays = json.loads(Path(path).read_text(encoding="utf-8"))["essays"]
    return [{"text": e["id"], "legs": [split(p[key]) for p in e["pairs"] for key in ("a", "b")]}
            for e in essays if len(e["pairs"]) == 4]


def leg_vectors(essays: list[dict], args, model, tokenizer, device) -> list[dict]:
    """Mean fragment vector of every leg, (legs, layers, dim); essays that do not tokenize are dropped."""
    kept = []
    for essay in essays:
        flat = [f for leg in essay["legs"] for f in leg]
        vectors = encode(flat, args.batch_size, model, tokenizer, device)
        if vectors is None:
            continue
        bounds = np.cumsum([0] + [len(leg) for leg in essay["legs"]])
        essay["vectors"] = np.stack([vectors[a:b].mean(axis=0) for a, b in zip(bounds[:-1], bounds[1:])])
        kept.append(essay)
    return kept


def unit(steps: np.ndarray) -> np.ndarray:
    return steps / np.linalg.norm(steps, axis=-1, keepdims=True)


def streams(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit transitions of the odd and the even stream: (layers, 3, dim) each."""
    odd, even = vectors[0::2], vectors[1::2]
    return (unit(np.diff(odd, axis=0)).transpose(1, 0, 2), unit(np.diff(even, axis=0)).transpose(1, 0, 2))


def cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.sum(first * second, axis=-1).mean(axis=-1)


def bootstrap(values: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    return values[rng.integers(0, len(values), size=(count, len(values)))].mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="欽定四書文=data/4_eight_legged_essays/qinding_regular100/legs.json")
    parser.add_argument("--controls", nargs="*", default=["Composed=data/4_eight_legged_essays/qinding_regular100/composed.json",
                                                           "Nonsense=data/4_eight_legged_essays/qinding_regular100/nonsense.json"],
                        help="label=path pairs of control essays built on the corpus, matched by essay id")
    parser.add_argument("--highlight", default="Chen Zilong=346")
    parser.add_argument("--prose", default="古文觀止=data/4_eight_legged_essays/qinding_regular100/guwen_pseudo.json",
                        help="label=path of prose pseudo-essays matched to the corpus (make_pseudo_essays.py)")
    parser.add_argument("--random-pairs", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--figure-layer", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-dir", default="results/baguwen_regular100/leg_streams")
    args = parser.parse_args()

    load_fonts()
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    rng = random.Random(args.seed)

    conditions: dict[str, list[dict]] = {}
    real_label, path = args.corpus.rsplit("=", 1)
    conditions[real_label] = load_essays(path)
    control_labels = []
    for spec in args.controls:
        label, path = spec.rsplit("=", 1)
        conditions[label] = load_essays(path)
        control_labels.append(label)
    prose_label, path = args.prose.rsplit("=", 1)
    conditions[prose_label] = load_essays(path)
    for label in conditions:
        conditions[label] = leg_vectors(conditions[label], args, model, tokenizer, device)
        for essay in conditions[label]:
            essay["odd"], essay["even"] = streams(essay["vectors"])
        print(f"{label}: {len(conditions[label])} essays")

    for label, essays in conditions.items():
        for essay in essays:
            prose = rng.sample([o for o in conditions[prose_label] if o is not essay],
                               min(args.random_pairs, len(conditions[prose_label]) - 1))
            same = rng.sample([o for o in essays if o is not essay], min(args.random_pairs, len(essays) - 1))
            essay["alignment"] = cosine(essay["odd"], essay["even"])
            essay["baseline"] = np.mean([cosine(essay["odd"], o["even"]) for o in prose], axis=0)
            essay["same_condition_baseline"] = np.mean([cosine(essay["odd"], o["even"]) for o in same], axis=0)
            essay["steps"] = np.sum(essay["odd"] * essay["even"], axis=-1)  # (layers, 3)

    L = args.figure_layer
    np_rng = np.random.default_rng(args.seed)
    diffs = {label: np.array([e["alignment"] - e["baseline"] for e in essays]) for label, essays in conditions.items()}
    summary = {}
    for label, essays in conditions.items():
        d = diffs[label]
        entry = {"essays": len(essays),
                 "alignment": np.mean([e["alignment"] for e in essays], axis=0).tolist(),
                 "baseline": np.mean([e["baseline"] for e in essays], axis=0).tolist(),
                 "same_condition_baseline": np.mean([e["same_condition_baseline"] for e in essays], axis=0).tolist(),
                 "difference": d.mean(axis=0).tolist(),
                 "difference_from_same_condition": np.mean([e["alignment"] - e["same_condition_baseline"]
                                                            for e in essays], axis=0).tolist(),
                 "by_transition_at_figure_layer": np.mean([e["steps"][L] for e in essays], axis=0).tolist(),
                 "difference_ci95_at_figure_layer": np.quantile(bootstrap(d[:, L], args.bootstrap, np_rng),
                                                                [0.025, 0.975]).tolist()}
        if label != prose_label:
            entry["vs_prose_mann_whitney_p"] = float(mannwhitneyu(d[:, L], diffs[prose_label][:, L],
                                                                  alternative="greater").pvalue)
            entry["vs_prose_cohens_d"] = cohens_d(d[:, L], diffs[prose_label][:, L])
        summary[label] = entry
    real = {e["text"]: e["alignment"][L] - e["baseline"][L] for e in conditions[real_label]}
    for label in control_labels:
        control = {e["text"]: e["alignment"][L] - e["baseline"][L] for e in conditions[label]}
        shared = sorted(set(real) & set(control))
        x, y = np.array([real[t] for t in shared]), np.array([control[t] for t in shared])
        summary[label]["vs_real_paired"] = {"essays": len(shared), "real_minus_control": float((x - y).mean()),
                                            "wilcoxon_p": float(wilcoxon(x, y).pvalue), "real_higher": int((x > y).sum())}
    highlight_label, highlight_id = args.highlight.rsplit("=", 1)
    highlighted = [e for e in conditions[real_label] if e["text"] == highlight_id]

    print(f"Layer {L}, essays as the unit (alignment minus prose baseline):")
    for label, entry in summary.items():
        line = (f"  {label}: {entry['essays']} essays, raw {entry['alignment'][L]:.3f}, prose baseline "
                f"{entry['baseline'][L]:.3f}, same-condition baseline {entry['same_condition_baseline'][L]:.3f}, "
                f"difference {entry['difference'][L]:.3f} CI [{entry['difference_ci95_at_figure_layer'][0]:.3f}, "
                f"{entry['difference_ci95_at_figure_layer'][1]:.3f}], by transition "
                + " ".join(f"{v:.3f}" for v in entry["by_transition_at_figure_layer"]))
        if "vs_prose_cohens_d" in entry:
            line += f"; vs prose d={entry['vs_prose_cohens_d']:.2f} p={entry['vs_prose_mann_whitney_p']:.1e}"
        if "vs_real_paired" in entry:
            v = entry["vs_real_paired"]
            line += (f"; real minus control {v['real_minus_control']:.3f} (real higher in {v['real_higher']} of "
                     f"{v['essays']}, Wilcoxon p={v['wilcoxon_p']:.1e})")
        print(line)
    if highlighted:
        print(f"{highlight_label}: difference at layer {L} {highlighted[0]['alignment'][L] - highlighted[0]['baseline'][L]:.3f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = len(summary[real_label]["difference"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    for label in conditions:
        axes[0].plot(range(layers), summary[label]["difference"], color=COLORS.get(label, "0.5"), linewidth=1.8,
                     label=f"{label} ({summary[label]['essays']} essays)")
    axes[0].axhline(0, color="0.6", linewidth=0.6)
    axes[0].set_xlabel("GujiBERT layer")
    axes[0].set_ylabel("Alignment minus random-pair baseline")
    axes[0].set_title("(a) Odd legs against even legs, by layer", fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    for position, label in enumerate(conditions):
        values = diffs[label][:, L]
        jitter = np.random.default_rng(position).uniform(-0.2, 0.2, len(values))
        marked = np.array([e["text"] == highlight_id for e in conditions[label]]) if label == real_label \
            else np.zeros(len(values), bool)
        axes[1].scatter(position + jitter[~marked], values[~marked], s=10, color=COLORS.get(label, "0.5"),
                        alpha=0.6, linewidths=0, zorder=2)
        axes[1].scatter(position + jitter[marked], values[marked], s=46, color=HIGHLIGHT_COLOR, edgecolors="white",
                        linewidths=0.8, zorder=4, label=highlight_label if marked.any() else None)
        low, high = summary[label]["difference_ci95_at_figure_layer"]
        axes[1].hlines(values.mean(), position - 0.3, position + 0.3, color="black", linewidth=1.5, zorder=3)
        axes[1].vlines(position + 0.34, low, high, color="black", linewidth=1.2, zorder=3)
    axes[1].axhline(0, color="0.6", linewidth=0.6)
    axes[1].set_xticks(range(len(conditions)))
    axes[1].set_xticklabels([f"{label}\n{summary[label]['essays']} essays" for label in conditions], fontsize=9)
    axes[1].set_title(f"(b) Every essay at layer {L}; bar = mean, whisker = 95% CI", fontsize=10, loc="left")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "leg_streams.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    (output_dir / "summary.json").write_text(json.dumps({
        "model": args.model, "prose": args.prose,
        "random_pairs": args.random_pairs, "bootstrap": args.bootstrap, "figure_layer": L, "seed": args.seed,
        "summary": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "essays.json").write_text(json.dumps({label: [
        {"text": e["text"], "legs": e["legs"], "alignment": np.round(e["alignment"], 4).tolist(),
         "baseline": np.round(e["baseline"], 4).tolist(),
         "same_condition_baseline": np.round(e["same_condition_baseline"], 4).tolist()} for e in essays]
        for label, essays in conditions.items()}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
