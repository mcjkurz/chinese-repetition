"""Self-similarity of a sliding window over an essay, as a curve over character position.

A window of --window characters (half to the left of a position, half to the
right) slides over the essay by --stride characters. Each window is encoded on
its own (one forward pass per window), and its self-similarity is the mean
cosine similarity over all pairs of distinct characters in it, using the
--layer embeddings centred on the mean over every character of every window.
The curve is plotted at the window's centre, with the essay's sections
(opening, legs, conclusion) marked along the x axis as in make_baguwen_figure.py.

As in the Mengzi comparison figure, panel (a) shows the original essay and the
later panels show each control set: its essays faintly and their mean in bold.
Control essays whose sections differ slightly in length from the original are
stretched section by section onto the original's sections before averaging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from qhchina import load_fonts
from scipy.ndimage import uniform_filter1d
from transformers import AutoModel, AutoTokenizer

from common import select_device
from make_baguwen_figure import PAIR_COLOR, SPAN_LABELS, label_bounds, section_bounds
from self_similarity_matrix import DEFAULT_MODEL

COLOR = "#1f4e8c"


def window_character_embeddings(text: str, window: int, stride: int, layer: int, batch_size: int,
                                model, tokenizer, device) -> tuple[list[int], np.ndarray]:
    """Window start positions and the layer embeddings of each window's characters, shape (windows, window, dim)."""
    starts = list(range(0, len(text) - window + 1, stride))
    states = []
    for batch_start in range(0, len(starts), batch_size):
        batch = [text[start:start + window] for start in starts[batch_start:batch_start + batch_size]]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        if encoded["input_ids"].shape[1] != window + 2:
            raise ValueError("Tokenization does not align with characters")
        with torch.no_grad():
            hidden = model(**encoded, output_hidden_states=True).hidden_states[layer]
        states.append(hidden[:, 1:-1].float().cpu().numpy())  # drop [CLS] and [SEP]
    return starts, np.concatenate(states)


def window_self_similarity(states: np.ndarray) -> np.ndarray:
    """Mean centred cosine similarity over distinct character pairs within each window."""
    centred = states - states.reshape(-1, states.shape[-1]).mean(axis=0)
    normalized = centred / np.linalg.norm(centred, axis=-1, keepdims=True)
    matrices = normalized @ normalized.transpose(0, 2, 1)
    size = states.shape[1]
    return (matrices.sum(axis=(1, 2)) - size) / (size * (size - 1))


def essay_curve(raw_text: str, args, model, tokenizer, device) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Window centres, smoothed window self-similarity, and section bounds of one essay."""
    text, bounds = section_bounds(raw_text)
    starts, states = window_character_embeddings(text, args.window, args.stride, args.layer, args.batch_size,
                                                 model, tokenizer, device)
    similarity = window_self_similarity(states)
    if args.smooth > 1:
        similarity = uniform_filter1d(similarity, args.smooth, mode="nearest")
    return np.array(starts) + (args.window - 1) / 2, similarity, bounds


def align_to_reference(x: np.ndarray, y: np.ndarray, bounds: list[int], reference_bounds: list[int],
                       grid: np.ndarray) -> np.ndarray:
    """Stretch each section of an essay onto the reference essay's section, then resample y on the grid."""
    warped = np.interp(x, bounds, reference_bounds)
    return np.interp(grid, warped, y, left=np.nan, right=np.nan)


def draw_sections(ax, bounds: list[int], ylabel: str = "Mean centered cosine\nwithin the window") -> None:
    spans = label_bounds(bounds)
    # Shade every other leg pair (legs 1-2, 5-6) so the pairs read as units.
    for pair, (start, end) in enumerate(zip(spans[1:9:2], spans[3:10:2])):
        if pair % 2 == 0:
            ax.axvspan(start - 0.5, end - 0.5, color=PAIR_COLOR, alpha=0.08, linewidth=0)
    for bound in spans[1:-1]:
        ax.axvline(bound - 0.5, color="0.6", linewidth=0.6)
    label_centres = [(start + end) / 2 for start, end in zip(spans[:-1], spans[1:])]
    ax.set_xticks(label_centres)
    ax.set_xticklabels(SPAN_LABELS, fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(0, bounds[-1])
    ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot(grid: np.ndarray, original: np.ndarray, groups: list[tuple[str, np.ndarray]], bounds: list[int],
         args, output: Path, ylabel: str = "Mean centered cosine\nwithin the window",
         legend_loc: str = "upper right") -> None:
    fig, axes = plt.subplots(1 + len(groups), 1, figsize=(11, 3.4 * (1 + len(groups))), sharey=args.shared_y)
    draw_sections(axes[0], bounds, ylabel)
    axes[0].plot(grid, original, color=COLOR, linewidth=1.8)
    axes[0].set_title("(a) Chen Zilong", fontsize=11, loc="left")
    for ax, (label, curves) in zip(axes[1:], groups):
        draw_sections(ax, bounds, ylabel)
        for curve in curves:
            ax.plot(grid, curve, color=COLOR, linewidth=0.6, alpha=0.2)
        ax.plot([], [], color=COLOR, linewidth=0.6, alpha=0.4, label=f"Individual essays (n={len(curves)})")
        ax.plot(grid, np.nanmean(curves, axis=0), color=COLOR, linewidth=1.8, label="Mean")
        ax.legend(frameon=False, fontsize=8, loc=legend_loc)
        ax.set_title(label, fontsize=11, loc="left")
    fig.tight_layout(h_pad=1.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="data/4_eight_legged_essays/baguwen_chen_zilong.txt")
    parser.add_argument(
        "--controls",
        nargs="*",
        default=["(b) Composed parallel essays=data/4_eight_legged_essays/baguwen_controls/sensical.json",
                 "(c) Repetitive nonsense=data/4_eight_legged_essays/baguwen_controls/nonsense.json",
                 "(d) Complete scramble=data/4_eight_legged_essays/baguwen_controls/scrambled.json"],
        help="label=path pairs; each JSON has a 'passages' list of {id, text}",
    )
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--smooth", type=int, default=15,
                        help="Width of the moving average applied to each curve, in positions (1 = none)")
    parser.add_argument("--shared-y", action="store_true", help="Give all panels the same y-axis range")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-dir", default="results/baguwen_window_similarity")
    args = parser.parse_args()

    load_fonts()
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    grid, original, bounds = essay_curve(Path(args.original).read_text(encoding="utf-8"), args,
                                         model, tokenizer, device)
    groups = []
    for spec in args.controls:
        label, path = spec.rsplit("=", 1)
        passages = json.loads(Path(path).read_text(encoding="utf-8"))["passages"]
        curves = []
        for passage in passages:
            x, y, passage_bounds = essay_curve(passage["text"], args, model, tokenizer, device)
            curves.append(align_to_reference(x, y, passage_bounds, bounds, grid))
        groups.append((label, np.array(curves)))
        print(f"{label}: {len(curves)} essays")

    output_dir = Path(args.output_dir)
    stem = f"window{args.window}_stride{args.stride}_layer{args.layer:02d}"
    plot(grid, original, groups, bounds, args, output_dir / f"{stem}.png")
    (output_dir / f"{stem}.json").write_text(json.dumps({
        "model": args.model, "original": args.original, "controls": args.controls, "window": args.window,
        "stride": args.stride, "layer": args.layer, "smooth": args.smooth, "section_bounds": bounds,
        "window_centres": grid.tolist(), "original_self_similarity": original.tolist(),
        "control_mean_self_similarity": {label: np.nanmean(curves, axis=0).tolist() for label, curves in groups},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
