"""Plot centered GujiBERT self-similarity matrices for an essay, and two control essays.

Panels share one colour scale; characters progress to the right and upward.
Each cell is averaged over a small neighbourhood (--smooth) so that block-level
structure is not drowned by character-level noise.
Thin lines mark the boundaries of the essay's sections (non-leg parts and legs),
and the section names are written along the bottom axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from qhchina import load_fonts
from scipy.ndimage import uniform_filter
from transformers import AutoModel, AutoTokenizer

from common import select_device
from self_similarity_matrix import (
    DEFAULT_MODEL,
    all_layer_character_embeddings,
    cosine_similarity_matrix,
    parse_leg_marked_text,
)

# Axis labels: the three opening parts (破題, 承題, 起講) share "Intro"; legs 1-2 are 起股, 3-4 中股,
# 5-6 後股, 7-8 束股; "Concl." is 大結.
PAIR_COLOR = "#5b4fc4"  # blue-purple outline of each leg pair
SPAN_LABELS = ["Intro", "Leg 1", "Leg 2", "Leg 3", "Leg 4", "Leg 5", "Leg 6", "Leg 7", "Leg 8", "Concl."]


def section_bounds(raw_text: str) -> tuple[str, list[int]]:
    """Text and section boundaries: one boundary after each line of the source file."""
    text, _, _ = parse_leg_marked_text(raw_text)
    bounds, position = [0], 0
    for line in raw_text.strip().splitlines():
        position += len(parse_leg_marked_text(line)[0])
        bounds.append(position)
    if bounds[-1] != len(text):
        raise ValueError("Section boundaries do not match the parsed text")
    return text, bounds


def label_bounds(bounds: list[int]) -> list[int]:
    """Boundaries of the labelled spans: the opening parts merged, then every leg, then the conclusion."""
    return [bounds[0]] + bounds[3:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="data/4_eight_legged_essays/baguwen_chen_zilong.txt")
    parser.add_argument("--composed", default="data/4_eight_legged_essays/baguwen_controls/sensical.json")
    parser.add_argument("--composed-id", default="composed_00")
    parser.add_argument("--nonsense", default="data/4_eight_legged_essays/baguwen_controls/nonsense.json")
    parser.add_argument("--nonsense-id", default="nonsense_00")
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cmap", default="Blues")
    parser.add_argument("--smooth", type=int, default=3,
                        help="Side of the averaging window in characters (1 = no smoothing)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output", default="results/baguwen_self_similarity/self_similarity.png",
                        help="Output file when a single layer is given")
    parser.add_argument("--output-dir", default="results/baguwen_self_similarity/fig5-by-layer",
                        help="Output folder (one layer_NN.png per layer) when several layers are given")
    args = parser.parse_args()

    def passage(path: str, passage_id: str) -> str:
        passages = json.loads(Path(path).read_text(encoding="utf-8"))["passages"]
        return next(item["text"] for item in passages if item["id"] == passage_id)

    original = Path(args.original).read_text(encoding="utf-8")
    panels = [
        ("Chen Zilong", original),
        ("Composed", passage(args.composed, args.composed_id)),
        ("Nonsense", passage(args.nonsense, args.nonsense_id)),
    ]

    load_fonts()
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    def layer_matrices(raw_text: str) -> tuple[dict[int, np.ndarray], list[int]]:
        """Centered cosine matrices for the requested layers, and the section bounds."""
        text, bounds = section_bounds(raw_text)
        chars, hidden = all_layer_character_embeddings(text, model, tokenizer, device)
        if len(chars) != len(text):
            raise ValueError("Tokenization does not align with characters")
        return {layer: cosine_similarity_matrix(hidden[layer], center=True) for layer in args.layers}, bounds

    matrices, all_bounds = [], []
    for _, raw_text in panels:
        layer_values, bounds = layer_matrices(raw_text)
        matrices.append(layer_values)
        all_bounds.append(bounds)

    for layer in args.layers:
        output = Path(args.output) if len(args.layers) == 1 else Path(args.output_dir) / f"layer_{layer:02d}.png"
        plot_layer(panels, [values[layer] for values in matrices], all_bounds, layer, args.cmap, args.dpi,
                   output, args.smooth)


def smooth_matrix(matrix: np.ndarray, window: int) -> np.ndarray:
    """Mean over a window x window neighbourhood of each cell, ignoring the diagonal."""
    values = matrix.copy()
    np.fill_diagonal(values, np.nan)
    if window <= 1:
        return values
    mask = np.isfinite(values).astype(float)
    total = uniform_filter(np.nan_to_num(values), window)
    count = uniform_filter(mask, window)
    return total / np.maximum(count, 1e-9)


def plot_layer(panels, raw_matrices, all_bounds, layer: int, cmap: str, dpi: int, output: Path, window: int) -> None:
    matrices = [smooth_matrix(matrix, window) for matrix in raw_matrices]

    finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
    # Negative similarities are drawn white, so the scale runs from zero to near the maximum.
    vmin, vmax = 0.0, float(np.percentile(finite, 99.5))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), gridspec_kw={"wspace": 0.06})
    for index, (ax, (title, _), matrix, bounds) in enumerate(zip(axes, panels, matrices, all_bounds)):
        image = ax.imshow(matrix, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        # A line after the opening and after every leg; labels sit at the centre of each span.
        spans = label_bounds(bounds)
        for bound in spans[1:-1]:
            ax.axvline(bound - 0.5, color="0.4", linewidth=0.6)
            ax.axhline(bound - 0.5, color="0.4", linewidth=0.6)
        # Outline the diagonal block of each leg pair (legs 1-2, 3-4, 5-6, 7-8).
        for start, end in zip(spans[1:9:2], spans[3:10:2]):
            ax.add_patch(Rectangle((start - 0.5, start - 0.5), end - start, end - start,
                                   fill=False, edgecolor=PAIR_COLOR, linewidth=2.0))
        centres = [(start + end) / 2 for start, end in zip(spans[:-1], spans[1:])]
        ax.set_xticks(centres)
        ax.set_xticklabels(SPAN_LABELS, fontsize=7, rotation=90)
        ax.set_yticks(centres)
        ax.set_yticklabels(SPAN_LABELS if index == 0 else [], fontsize=7)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
    label = "Centered cosine similarity" + (f"\n(mean over {window} × {window} pairs)" if window > 1 else "")
    fig.colorbar(image, ax=axes, label=label, fraction=0.015, pad=0.01)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
