"""Transition alignment of the two strings of legs in an eight-legged essay.

The legs are read as two lines of a couplet: the first members of the pairs
(legs 1, 3, 5, 7) and the second members (legs 2, 4, 6, 8). Each leg is the
mean of its characters' embeddings (the whole essay encoded in context,
punctuation left out), as in baguwen_leg_space.py; the opening sections and
the conclusion are not used. As in transition_analysis.py, the transitions of
each string are the differences between consecutive legs (leg 3 - leg 1,
leg 5 - leg 3, leg 7 - leg 5, and leg 4 - leg 2, ...), and the alignment of an
essay is the mean cosine between corresponding transitions of the two strings.

Three baselines are computed at every layer:

- position-matched random pairs: the transition in one slot of an essay
  compared with the transition of the other string in the same slot of a
  different essay of the same set (for Chen Zilong: of the composed essays),
  as in the couplet experiment;
- mismatched slots: the same essay, but transitions in different slots;
- the scrambled essays, whose legs have the same lengths and positions but no
  wording in common.

Writes a figure of alignment by layer (raw and baseline-subtracted) and a JSON
file with all values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, AutoTokenizer

from common import select_device
from lm_scoring_utils import PUNCTUATION
from make_baguwen_figure import section_bounds
from self_similarity_matrix import DEFAULT_MODEL, all_layer_character_embeddings

ODD_LEGS = [3, 5, 7, 9]  # section indices of legs 1, 3, 5, 7
EVEN_LEGS = [4, 6, 8, 10]  # legs 2, 4, 6, 8
COLORS = {"Chen Zilong": "#08306b", "Composed": "#2b7bba", "Nonsense": "#79b4dd", "Scramble": "#b0b0b8"}


def leg_transitions(raw_text: str, model, tokenizer, device) -> tuple[np.ndarray, np.ndarray]:
    """Unit transition vectors of the odd and even strings, each of shape (layers, 3, dim)."""
    text, bounds = section_bounds(raw_text)
    chars, hidden = all_layer_character_embeddings(text, model, tokenizer, device)
    if len(chars) != len(text):
        raise ValueError("Tokenization does not align with characters")
    means = np.stack([hidden[:, [i for i in range(start, end) if text[i] not in PUNCTUATION]].mean(axis=1)
                      for start, end in zip(bounds[:-1], bounds[1:])], axis=1)
    strings = []
    for legs in (ODD_LEGS, EVEN_LEGS):
        steps = np.diff(means[:, legs], axis=1)
        strings.append(steps / np.linalg.norm(steps, axis=-1, keepdims=True))
    return strings[0], strings[1]


def alignment(odd: np.ndarray, even: np.ndarray) -> np.ndarray:
    """Mean cosine between corresponding transitions, per layer: odd and even have shape (layers, slots, dim)."""
    return np.sum(odd * even, axis=-1).mean(axis=-1)


def mismatched(odd: np.ndarray, even: np.ndarray) -> np.ndarray:
    """Mean cosine between transitions in different slots, per layer."""
    cosines = odd @ even.transpose(0, 2, 1)
    off = ~np.eye(cosines.shape[-1], dtype=bool)
    return cosines[:, off].mean(axis=-1)


def random_pairs(odd: np.ndarray, pool_even: list[np.ndarray]) -> np.ndarray:
    """Mean cosine between an essay's odd transitions and the same slots of other essays' even ones, per layer."""
    return np.mean([alignment(odd, even) for even in pool_even], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="data/4_eight_legged_essays/baguwen_chen_zilong.txt")
    parser.add_argument(
        "--controls",
        nargs="*",
        default=["Composed=data/4_eight_legged_essays/baguwen_controls/sensical.json",
                 "Nonsense=data/4_eight_legged_essays/baguwen_controls/nonsense.json",
                 "Scramble=data/4_eight_legged_essays/baguwen_controls/scrambled.json"],
        help="label=path pairs; each JSON has a 'passages' list of {id, text}",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-dir", default="results/baguwen_leg_alignment")
    args = parser.parse_args()

    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    chen_odd, chen_even = leg_transitions(Path(args.original).read_text(encoding="utf-8"), model, tokenizer, device)
    sets = {}
    for spec in args.controls:
        label, path = spec.rsplit("=", 1)
        passages = json.loads(Path(path).read_text(encoding="utf-8"))["passages"]
        sets[label] = [leg_transitions(passage["text"], model, tokenizer, device) for passage in passages]

    results = {"Chen Zilong": {
        "alignment": alignment(chen_odd, chen_even).tolist(),
        "slot_alignment": np.sum(chen_odd * chen_even, axis=-1).tolist(),
        "random_pair_baseline": random_pairs(chen_odd, [even for _, even in sets["Composed"]]).tolist(),
        "mismatched_baseline": mismatched(chen_odd, chen_even).tolist(),
    }}
    for label, essays in sets.items():
        own = np.array([alignment(odd, even) for odd, even in essays])
        baseline = np.array([random_pairs(odd, [even for j, (_, even) in enumerate(essays) if j != i])
                             for i, (odd, _) in enumerate(essays)])
        results[label] = {
            "alignment": own.mean(axis=0).tolist(), "alignment_sd": own.std(axis=0).tolist(),
            "random_pair_baseline": baseline.mean(axis=0).tolist(),
            "mismatched_baseline": np.mean([mismatched(odd, even) for odd, even in essays], axis=0).tolist(),
            "share_above_baseline": np.mean(own > baseline, axis=0).tolist(),
        }

    layers = np.arange(len(results["Chen Zilong"]["alignment"]))
    print("layer  " + "  ".join(f"{name:>30}" for name in results))
    print("       " + "  ".join(f"{'aligned / random / mismatched':>30}" for _ in results))
    for layer in layers:
        print(f"{layer:5d}  " + "  ".join(
            f"{r['alignment'][layer]:10.3f}{r['random_pair_baseline'][layer]:10.3f}{r['mismatched_baseline'][layer]:10.3f}"
            for r in results.values()))
    print("Chen Zilong, per slot (1-3 vs 2-4, 3-5 vs 4-6, 5-7 vs 6-8):")
    for layer in layers:
        print(f"{layer:5d}  " + "  ".join(f"{value:+.3f}" for value in results["Chen Zilong"]["slot_alignment"][layer]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for label, r in results.items():
        aligned, baseline = np.array(r["alignment"]), np.array(r["random_pair_baseline"])
        width = 2.2 if label == "Chen Zilong" else 1.4
        axes[0].plot(layers, aligned, color=COLORS[label], linewidth=width, label=label)
        axes[0].plot(layers, baseline, color=COLORS[label], linewidth=1, linestyle=":")
        axes[1].plot(layers, aligned - baseline, color=COLORS[label], linewidth=width, label=label)
    axes[0].set_title("(a) Alignment (solid) and random-pair baseline (dotted)", fontsize=10, loc="left")
    axes[1].set_title("(b) Alignment minus random-pair baseline", fontsize=10, loc="left")
    axes[1].axhline(0, color="0.6", linewidth=0.6)
    for ax in axes:
        ax.set_xlabel("GujiBERT layer")
        ax.set_ylabel("Mean cosine of corresponding transitions")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "leg_alignment.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    (output_dir / "leg_alignment.json").write_text(json.dumps({
        "model": args.model, "original": args.original, "controls": args.controls,
        "odd_legs": [1, 3, 5, 7], "even_legs": [2, 4, 6, 8], "results": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output_dir / 'leg_alignment.png'}")


if __name__ == "__main__":
    main()
