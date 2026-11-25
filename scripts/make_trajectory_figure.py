"""Draw the 2 x 2 couplet-trajectory figure (Figure 3).

Rows: pentasyllabic, heptasyllabic. Columns: parallel (inner), non-parallel.
Each couplet is encoded with GujiBERT as "line1，line2。" (the same context as
transition_analysis.py); its final-layer character embeddings are projected
into the couplet's own three principal components for display. The transition
alignment printed for each couplet is computed in the full 768-dimensional
space, as in the paper's statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from qhchina import load_fonts

import palette as P
from transformers import AutoModel, AutoTokenizer

from transition_analysis import (
    DEFAULT_MODEL,
    couplet_context_and_targets,
    get_character_embeddings_batch,
    transition_alignment,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "figures" / "fig3-couplet-trajectories.png"
LAYER = 12

COUPLETS = [
    {"panel": "a", "meter": "Pentasyllabic", "kind": "parallel",
     "text": "明月松間照，清泉石上流", "source": "Wang Wei",
     "translation": ("The bright moon shines among the pines;",
                     "the clear spring flows over the stones.")},
    {"panel": "b", "meter": "Pentasyllabic", "kind": "non-parallel",
     "text": "欲投人處宿，隔水問樵夫", "source": "Wang Wei",
     "translation": ("Wanting to find a place in the mountains to stay the night,",
                     "I ask a woodcutter across the stream.")},
    {"panel": "c", "meter": "Heptasyllabic", "kind": "parallel",
     "text": "萬里悲秋常作客，百年多病獨登臺", "source": "Du Fu",
     "translation": ("Ten thousand miles from home, grieving at autumn, ever a traveler;",
                     "a hundred years of illness, alone I climb the terrace.")},
    {"panel": "d", "meter": "Heptasyllabic", "kind": "non-parallel",
     "text": "劍外忽傳收薊北，初聞涕淚滿衣裳", "source": "Du Fu",
     "translation": ("Beyond Sword Gate, the news suddenly spreads that Jibei has been taken back;",
                     "when I first hear it, tears fill my clothes.")},
]

# Light shade for the first line, dark shade for the second, in the article's palette (palette.py): blue
# for real texts with the pattern, bluish green for real texts without it.
PALETTE = {
    "parallel": (P.REAL_LIGHT, P.REAL),
    "non-parallel": (P.WITHOUT_PATTERN_LIGHT, P.WITHOUT_PATTERN_DARK),
}
INK = "#2b2b2b"
MUTED = "#8a8a8a"


def per_couplet_pca(line1: np.ndarray, line2: np.ndarray, dims: int = 3):
    """Project both lines into the couplet's own principal subspace."""
    combined = np.vstack([line1, line2])
    centered = combined - combined.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:dims].T
    variance = singular_values**2 / np.sum(singular_values**2)
    n = len(line1)
    return projected[:n], projected[n:], variance[:dims]


def encode(couplets: list[dict]) -> None:
    """Attach final-layer embeddings, PCA coordinates, and alignment to each couplet."""
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModel.from_pretrained(DEFAULT_MODEL, output_hidden_states=True).eval()
    for item in couplets:
        line1, line2 = item["text"].split("，")
        context, targets = couplet_context_and_targets(line1, line2, True)
        embeddings = get_character_embeddings_batch(
            [context], [targets], tokenizer, model, torch.device("cpu")
        )[0][LAYER]
        e1, e2 = embeddings[: len(line1)], embeddings[len(line1):]
        item["line1"], item["line2"] = line1, line2
        item["alignment_768"] = transition_alignment(e1, e2)["mean_alignment"]
        item["p1"], item["p2"], item["variance"] = per_couplet_pca(e1, e2)


def orient(p1: np.ndarray, p2: np.ndarray):
    """Flip axis signs so that each line runs left to right along PC1."""
    signs = np.ones(3)
    if np.mean(p1[-1] - p1[0] + p2[-1] - p2[0]) < 0:
        signs[0] = -1
    if np.mean(np.r_[p1[:, 2], p2[:, 2]][:1]) < 0:
        signs[2] = -1
    return p1 * signs, p2 * signs


def best_view(points: np.ndarray, box_aspect=(1.25, 1, 0.85)) -> tuple[int, int]:
    """Choose the orthographic view that keeps projected characters furthest apart."""
    span = points.max(axis=0) - points.min(axis=0)
    scaled = (points - points.min(axis=0)) / np.where(span == 0, 1, span) * box_aspect
    best, best_score = (20, -62), -np.inf
    # Search a moderate range around matplotlib's default so the panels look alike.
    for elev in range(14, 31, 4):
        for azim in range(-96, -23, 4):
            el, az = np.radians(elev), np.radians(azim)
            x = -np.sin(az) * scaled[:, 0] + np.cos(az) * scaled[:, 1]
            y = (-np.sin(el) * np.cos(az) * scaled[:, 0]
                 - np.sin(el) * np.sin(az) * scaled[:, 1]
                 + np.cos(el) * scaled[:, 2])
            screen = np.c_[x, y]
            distances = np.linalg.norm(screen[:, None] - screen[None], axis=-1)
            distances = distances[np.triu_indices(len(screen), 1)]
            # Smallest gaps dominate; the mean breaks ties among similar views.
            score = np.sort(distances)[:3].mean()
            if score > best_score:
                best, best_score = (elev, azim), score
    return best


def draw_panel(ax, item: dict) -> None:
    light, dark = PALETTE[item["kind"]]
    p1, p2 = orient(item["p1"], item["p2"])

    for coords, line, color in ((p1, item["line1"], light), (p2, item["line2"], dark)):
        ax.plot(*coords.T, color=color, linewidth=2.4, solid_capstyle="round", zorder=2)
        ax.scatter(*coords.T, s=430, color=color, edgecolor="white", linewidth=1.2,
                   depthshade=False, zorder=3)
        text_color = "white" if color == dark else INK
        for (x, y, z), char in zip(coords, line):
            ax.text(x, y, z, char, ha="center", va="center", fontsize=11.5,
                    color=text_color, zorder=4)

    ax.set_proj_type("ortho")
    ax.view_init(*best_view(np.vstack([p1, p2])))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.975, 0.975, 0.975, 1.0))
        axis.pane.set_edgecolor("#d9d9d9")
        axis._axinfo["grid"].update(color="#e3e3e3", linewidth=0.6)
        # PCA coordinates have no meaningful units, so ticks carry no labels.
        axis.set_ticklabels([])
        axis._axinfo["tick"].update(inward_factor=0, outward_factor=0)
    variance = item["variance"] * 100
    ax.set_xlabel(f"PC1 ({variance[0]:.1f}%)", fontsize=9, color=MUTED, labelpad=-12)
    ax.set_ylabel(f"PC2 ({variance[1]:.1f}%)", fontsize=9, color=MUTED, labelpad=-12)
    ax.set_zlabel(f"PC3 ({variance[2]:.1f}%)", fontsize=9, color=MUTED, labelpad=-12)
    # Zoomed so that each box fills its axes and the panels sit close together.
    ax.set_box_aspect((1.25, 1, 0.85), zoom=1.12)

    header = (
        (f"({item['panel']}) {item['line1']}，{item['line2']}", 12.5, INK, "normal"),
        (item["translation"][0], 8, INK, "italic"),
        (item["translation"][1], 8, INK, "italic"),
    )
    # Poet, kind and alignment are left to the caption and the colours; main() still prints each alignment.
    for y, (text, size, color, style) in zip((1.02, 0.965, 0.935), header):
        ax.text2D(0.5, y, text, transform=ax.transAxes, ha="center", va="center",
                  fontsize=size, color=color, fontstyle=style)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    load_fonts()
    couplets = [dict(item) for item in COUPLETS]
    encode(couplets)

    fig = plt.figure(figsize=(9.4, 9.2))
    for idx, item in enumerate(couplets):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        draw_panel(ax, item)

    for row, meter in enumerate(("Pentasyllabic", "Heptasyllabic")):
        fig.text(0.035, 0.73 - row * 0.465, meter, rotation=90, ha="center",
                 va="center", fontsize=13, color=INK)

    handles = [
        Line2D([], [], color=MUTED, marker="o", markerfacecolor="#c9c9c9",
               markeredgecolor="white", markersize=11, linewidth=2.2, label="first line"),
        Line2D([], [], color=INK, marker="o", markerfacecolor="#555555",
               markeredgecolor="white", markersize=11, linewidth=2.2, label="second line"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.93, bottom=0.05, wspace=-0.02, hspace=0.12)
    fig.savefig(args.output, dpi=300, facecolor="white")
    plt.close(fig)

    summary = [
        {k: item[k] for k in ("panel", "text", "kind", "source")}
        | {"alignment_768_layer12": round(float(item["alignment_768"]), 3),
           "pca_variance": [round(float(v), 3) for v in item["variance"]]}
        for item in couplets
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
