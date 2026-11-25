"""Compare GujiGPT with Qwen3 base models on the Mengzi passage and its controls.

Qwen models score the simplified text (OpenCC t2s), since they split probability
between traditional and simplified forms. A Qwen token can cover several
characters; its surprisal is kept as one point placed midway between them.
Qwen curves are cached in --curves and only rescored with --rescore.

Writes the figure model_comparison_gujigpt_qwen3_8b.png (a GujiGPT panel above
a Qwen3-8B one, with one line and one row of characters per condition) and the
table repetition_layers.json/.md (first and second member of each repeated pair).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import opencc
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import select_device
from qhchina import load_fonts

import palette as P
from sequence_surprisal import write_json


N = 63
# First and second member of each repeated pair (1-based character spans, inclusive).
LAYER_PAIRS = {
    "inner, first half": ((2, 6), (8, 15)),
    "inner, second half": ((33, 39), (41, 47)),
    "conclusion": ((17, 32), (49, 63)),
    "outer (halves)": ((2, 32), (33, 63)),
}
TEXTS = {
    "mengzi": "data/1_mengzi_prose/mengzi_surprisal.txt",
    "composed": "data/1_mengzi_prose/surprisal_controls/sensical.json",
    "nonsense": "data/1_mengzi_prose/surprisal_controls/items_random_composed_frame.json",
    "scrambled": "data/1_mengzi_prose/surprisal_controls/scrambled.json",
}
GUJIGPT_RESULTS = {
    "mengzi": "results/surprisal/mengzi_surprisal.json",
    "composed": "results/surprisal/controls_sensical.json",
    "nonsense": "results/surprisal/controls_nonsense_items.json",
    "scrambled": "results/surprisal/controls_scrambled.json",
}
QWEN = ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base", "Qwen/Qwen3-8B-Base"]
LABELS = {"GujiGPT": "GujiGPT", **{q: q.removeprefix("Qwen/").removesuffix("-Base") for q in QWEN}}
TITLES = {
    "mengzi": "Mengzi 6A10",
    "composed": "Composed controls (mean of 100 passages)",
    "nonsense": "Nonsense controls (mean of 100 passages)",
    "scrambled": "Shuffled controls (mean of 100 passages)",
}


def load_texts() -> dict[str, list[str]]:
    texts = {"mengzi": [Path(TEXTS["mengzi"]).read_text(encoding="utf-8").strip()]}
    for condition in ("composed", "nonsense", "scrambled"):
        passages = json.loads(Path(TEXTS[condition]).read_text(encoding="utf-8"))["passages"]
        texts[condition] = [p["text"] for p in passages]
    return texts


def gujigpt_curves() -> dict[str, list[dict[float, float]]]:
    """Per-character curves as {position: surprisal}, from the existing GujiGPT results."""
    curves = {}
    for condition, path in GUJIGPT_RESULTS.items():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        passages = [data] if condition == "mengzi" else data["passages"]
        curves[condition] = [
            {p["position"]: p["surprisal_nats"] for p in passage["points"] if p["surprisal_nats"] is not None}
            for passage in passages
        ]
    return curves


def score_qwen(model_name: str, texts: dict[str, list[str]], device) -> dict[str, list[list[list[float]]]]:
    """Return, per condition and passage, [x, surprisal] pairs with x at each token's midpoint."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device).eval()
    converter = opencc.OpenCC("t2s")
    curves = {}
    for condition, items in texts.items():
        curves[condition] = []
        for text in tqdm(items, desc=f"{LABELS[model_name]} {condition}"):
            text = converter.convert(text)
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
            offsets = [tuple(span) for span in encoded.pop("offset_mapping")[0].tolist()]
            ids = encoded["input_ids"].to(device)
            with torch.no_grad():
                log_probs = torch.log_softmax(model(ids).logits[0].float(), dim=-1)
            # Byte-level tokens of one character share its span and are summed.
            spans: dict[tuple[int, int], float] = {}
            for i, span in enumerate(offsets[1:], start=1):
                spans[span] = spans.get(span, 0.0) - log_probs[i - 1, ids[0, i]].item()
            spans.pop(offsets[0], None)
            curves[condition].append([[(start + 1 + end) / 2, value] for (start, end), value in sorted(spans.items())])
    return curves


def load_curves(path: Path, texts, device, rescore: bool) -> dict[str, dict[str, list[dict[float, float]]]]:
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() and not rescore else {}
    for model_name in QWEN:
        # Score only the conditions missing from the cache, so adding a condition keeps the others.
        missing = {c: items for c, items in texts.items() if c not in stored.get(model_name, {})}
        if missing:
            stored.setdefault(model_name, {}).update(score_qwen(model_name, missing, device))
            write_json(path, stored)
    curves = {"GujiGPT": gujigpt_curves()}
    for model_name in QWEN:
        curves[model_name] = {
            condition: [{x: value for x, value in passage} for passage in passages]
            for condition, passages in stored[model_name].items()
        }
    return curves


def mean_curve(passages: list[dict[float, float]]) -> tuple[list[float], list[float]]:
    """Mean at each x over the passages that have a point there (merged tokens sit at half positions); an x where
    fewer than a tenth of the passages have a point is left out, since its mean would rest on a few passages."""
    counts = {}
    for passage in passages:
        for x in passage:
            counts[x] = counts.get(x, 0) + 1
    xs = sorted(x for x, n in counts.items() if n >= max(1, len(passages) / 10))
    return xs, [float(np.mean([p[x] for p in passages if x in p])) for x in xs]


def to_grid(curve: dict[float, float]) -> np.ndarray:
    """One value per character: a merged token's surprisal is split evenly over its characters."""
    grid = np.full(N + 1, np.nan)
    end = N  # walk back from the last character, where every passage ends, to recover token widths
    for x in sorted(curve, reverse=True):
        width = int(round(2 * (end - x) + 1))
        grid[end - width + 1 : end + 1] = curve[x] / width
        end -= width
    return grid[1:]


def style_x(axis, chars: list[str], top: bool) -> None:
    positions = np.arange(1, N + 1)
    axis.set_xlim(0.5, N + 0.5)
    axis.set_xticks(positions)
    axis.set_xticklabels(chars, fontsize=12)
    if top:
        ticks = [1, *range(5, N + 1, 5)]
        secondary = axis.secondary_xaxis("top")
        secondary.set_xticks(ticks)
        secondary.set_xticklabels([str(t) for t in ticks], fontsize=12)


def curly_brace(axis, start: float, end: float, y: float, depth: float, label: str = "", gap: float = 0.03) -> None:
    """Draw a downward curly brace from position start to end at axes height y (x in data, y in axes units)."""
    from matplotlib.transforms import blended_transform_factory

    transform = blended_transform_factory(axis.transData, axis.transAxes)
    t = np.linspace(0, 0.5, 200)
    half = 1 / (1 + np.exp(-60 * t)) + 1 / (1 + np.exp(-60 * (t - 0.5)))
    half = (half - half.min()) / (half.max() - half.min())
    shape = np.concatenate([half, half[-2::-1]])
    xs = np.linspace(start - 0.4, end + 0.4, len(shape))
    axis.plot(xs, y - depth * shape, color="#333333", linewidth=1, transform=transform, clip_on=False)
    if label:
        axis.text((start + end) / 2, y - depth - gap, label, transform=transform, ha="center", va="top",
                  fontsize=10, color="#333333")


def draw_repetition_layers(axis, top: float = -0.31, scale: float = 1.0) -> None:
    """Mark the layers of repetition under the axis: parallel clauses within each half, the conclusion
    that recombines their two items, and the two halves. Offsets are in axes units, multiplied by scale."""
    from matplotlib.transforms import blended_transform_factory

    transform = blended_transform_factory(axis.transData, axis.transAxes)
    rows = [
        ("Inner layer", [(1, 6), (8, 15), (33, 39), (41, 47)], ["", "", "", ""]),
        ("Conclusion", [(17, 32), (49, 63)], ["", ""]),
        ("Outer layer", [(1, 32), (33, 63)], ["first half", "second half"]),
    ]
    for row, (name, spans, labels) in enumerate(rows):
        y = top - row * 0.19 * scale
        for (start, end), label in zip(spans, labels):
            curly_brace(axis, start, end, y, depth=0.07 * scale, label=label, gap=0.03 * scale)
        axis.text(0.3, y - 0.035 * scale, name, transform=transform, ha="right", va="center", fontsize=10,
                  color="#333333")


# Conditions drawn in the figure: the scrambled passages show the decline across the passage without repetition.
FIGURE_CONDITIONS = ("mengzi", "composed", "nonsense", "scrambled")


def plot_model_panels(curves, texts, output: Path, models: list[str]) -> None:
    """One panel per model, each with a line per condition of FIGURE_CONDITIONS; under the last panel, a
    row of characters per condition and the layers of repetition."""
    from matplotlib.transforms import blended_transform_factory

    # The article's palette (palette.py); the rows of characters in darker variants, for legibility.
    colors = {"mengzi": P.REAL, "composed": P.COMPOSED, "nonsense": P.NONSENSE, "scrambled": P.SHUFFLED}
    text_colors = {"mengzi": P.TEXT["real"], "composed": P.TEXT["composed"], "nonsense": P.TEXT["nonsense"],
                   "scrambled": P.TEXT["shuffled"]}
    names = {"mengzi": "Mengzi 6A10", "composed": "Composed (mean of 100)", "nonsense": "Nonsense (mean of 100)",
             "scrambled": "Shuffled (mean of 100)"}
    row_names = {"mengzi": "Mengzi", "composed": "Composed (e.g.)", "nonsense": "Nonsense (e.g.)",
                 "scrambled": "Shuffled (e.g.)"}
    extra_rows = len(FIGURE_CONDITIONS) - 3  # the offsets below were set for three rows of characters
    panel_height, gap, below = 2.2, 0.55, 2.9 + 0.24 * extra_rows  # inches: plot box, space between boxes, rows and braces below
    k = 3.2 / panel_height  # offsets below were set for a 3.2-inch box; keep them constant in inches
    height = len(models) * panel_height + (len(models) - 1) * gap + 0.75 + below
    fig, axes = plt.subplots(len(models), 1, figsize=(12, height), sharex=True, sharey=True, squeeze=False)
    axes = axes[:, 0]
    # Fixed axes boxes: the rows and braces below are placed in axes units and need room.
    fig.subplots_adjust(left=0.08, right=0.98, top=1 - 0.75 / height, bottom=below / height, hspace=gap / panel_height)
    for index, (axis, model) in enumerate(zip(axes, models)):
        for condition in FIGURE_CONDITIONS:
            if condition == "scrambled":
                # Merged tokens fall at different places in every scramble, so midpoints would alternate with
                # whole positions in the mean; split each merged token evenly over its characters instead.
                xs = np.arange(1, N + 1)
                mean = np.nanmean(np.vstack([to_grid(p) for p in curves[model][condition]]), axis=0)
            else:
                xs, mean = mean_curve(curves[model][condition])
            axis.plot(xs, mean, color=colors[condition], linewidth=2, label=names[condition])
        axis.set_title(LABELS[model], fontsize=14, loc="left", pad=18 if index == 0 else 6)
        axis.set_ylabel("Surprisal (nats)", fontsize=11)
        axis.grid(alpha=0.25)
        axis.yaxis.set_major_locator(plt.MultipleLocator(2))
        style_x(axis, [""] * N, top=index == 0)
    # One row of characters per condition, in the condition's colour (control rows show one example passage).
    axis = axes[-1]
    transform = blended_transform_factory(axis.transData, axis.transAxes)
    for row, condition in enumerate(FIGURE_CONDITIONS):
        y = (-0.04 - row * 0.075) * k
        for position, char in enumerate(texts[condition][0], start=1):
            axis.text(position, y, char, transform=transform, ha="center", va="top", fontsize=12,
                      color=text_colors[condition])
        axis.text(0.3, y, row_names[condition], transform=transform, ha="right", va="top", fontsize=10,
                  color=text_colors[condition])
    draw_repetition_layers(axis, top=(-0.30 - 0.075 * extra_rows) * k, scale=0.6 * k)
    # In the last panel: GujiGPT peaks under the upper right corner of the first one.
    axes[-1].legend(loc="upper right", fontsize=9)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")

def layer_comparison(curves, output: Path) -> None:
    """Mean surprisal of the first and second member of each repeated pair, per model and condition.

    Means are taken per passage over the characters of each span (punctuation included; position 1 is
    unscored), then averaged over passages. Also records how many passages show a lower second member.
    """
    rows = {}
    for model in ["GujiGPT", *QWEN]:
        for condition in TITLES:
            grid = np.vstack([to_grid(p) for p in curves[model][condition]])
            rows[f"{LABELS[model]} / {condition}"] = {}
            for name, ((a, b), (c, d)) in LAYER_PAIRS.items():
                first = np.nanmean(grid[:, a - 1 : b], axis=1)
                second = np.nanmean(grid[:, c - 1 : d], axis=1)
                rows[f"{LABELS[model]} / {condition}"][name] = {
                    "first": float(first.mean()),
                    "second": float(second.mean()),
                    "second_lower": int((second < first).sum()),
                    "n": len(first),
                }
    write_json(output, {"pairs": {k: [list(v[0]), list(v[1])] for k, v in LAYER_PAIRS.items()}, "rows": rows})
    print(f"Saved: {output}")
    lines = [
        "| Model / condition | " + " | ".join(LAYER_PAIRS) + " |",
        "|:--|" + ":--:|" * len(LAYER_PAIRS),
    ]
    for row, values in rows.items():
        cells = [f"{v['first']:.1f} → {v['second']:.1f} ({v['second_lower']}/{v['n']})" for v in values.values()]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    markdown = output.with_suffix(".md")
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {markdown}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--curves", default="results/surprisal/qwen_surprisal_curves.json")
    parser.add_argument("--output-dir", default="results/surprisal")
    parser.add_argument("--rescore", action="store_true", help="Rescore the Qwen models even if cached.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    load_fonts()
    texts = load_texts()
    curves = load_curves(Path(args.curves), texts, select_device(args.device), args.rescore)
    output_dir = Path(args.output_dir)
    plot_model_panels(curves, texts, output_dir / "model_comparison_gujigpt_qwen3_8b.png",
                      ["GujiGPT", "Qwen/Qwen3-8B-Base"])
    layer_comparison(curves, output_dir / "repetition_layers.json")


if __name__ == "__main__":
    main()
