"""Compare Mengzi surprisal or entropy with template-matched control passages.

The reference (a single sequence_surprisal.py result) is drawn as one curve.
Each control group (a score_passages.py result) is drawn as faint per-passage
points with the per-position mean on top. Each panel scales its own y-axis
unless --shared-y is given. The phrase 舍魚而取熊掌者也。 (positions 24-32) and
its repetition 舍生而取義者也。 (56-63) are shaded in every panel. The mean of the chosen --metric is summarized by
role (content, frame, punctuation) and half of the passage. The "excess"
metric is surprisal minus entropy: how far the observed surprisal exceeded the
surprisal the model expected. Output file names end in _surprisal, _entropy or
_excess according to the metric.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qhchina import load_fonts
from sequence_surprisal import (
    draw_surprisal,
    set_position_axes,
    style_surprisal_axis,
    write_json,
)


DEFAULT_REFERENCE = "results/surprisal/mengzi_surprisal.json"
DEFAULT_REFERENCE_LABEL = "(a) Mengzi 6A10"
DEFAULT_GROUPS = [
    "results/surprisal/controls_sensical.json",
    "results/surprisal/controls_nonsense.json",
]
DEFAULT_GROUP_LABELS = [
    "(b) Composed parallel passages",
    "(c) Repetitive nonsense",
]
DEFAULT_OUTPUT_PNG = "results/surprisal/mengzi_controls_comparison_{metric}.png"
DEFAULT_OUTPUT_JSON = "results/surprisal/mengzi_controls_summary_{metric}.json"
METRICS = ("surprisal", "entropy", "excess")

PUNCTUATION = set("，；。、！？：")
# 1-based positions of the substitutable "content" slots in the Mengzi template:
# 魚 / 熊掌 / 魚 / 熊掌 in the first half and 生 / 義 / 生 / 義 in the second.
MENGZI_CONTENT_POSITIONS = {1, 8, 9, 25, 28, 29, 33, 41, 57, 60}
MENGZI_HALF_BOUNDARY = 33
COLOR = "#2878a5"
# 1-based inclusive position ranges shaded in every panel: the phrase
# 舍魚而取熊掌者也。 and its repetition 舍生而取義者也。
HIGHLIGHT_RANGES = [(24, 32), (56, 63)]
HIGHLIGHT_COLOR = "#ffe680"


def classify(points: list[dict[str, object]]) -> list[str]:
    roles = []
    for point in points:
        if point["token"] in PUNCTUATION:
            roles.append("punctuation")
        elif point["position"] in MENGZI_CONTENT_POSITIONS:
            roles.append("content")
        else:
            roles.append("frame")
    return roles


def role_means(
    points: list[dict[str, object]],
    metric: str,
) -> dict[str, dict[str, float | None]]:
    key = f"{metric}_nats"
    roles = classify(points)
    means: dict[str, dict[str, float | None]] = {}
    for half, keep in (
        ("first_half", lambda p: p < MENGZI_HALF_BOUNDARY),
        ("second_half", lambda p: p >= MENGZI_HALF_BOUNDARY),
        ("all", lambda p: True),
    ):
        means[half] = {}
        for role in ("content", "frame", "punctuation"):
            values = [
                point[key]
                for point, point_role in zip(points, roles)
                if point_role == role
                and keep(point["position"])
                and point[key] is not None
            ]
            means[half][role] = float(np.mean(values)) if values else None
    return means


def add_excess(points: list[dict[str, object]]) -> None:
    """Store surprisal minus entropy on each point as excess_nats."""
    for point in points:
        surprisal, entropy = point["surprisal_nats"], point["entropy_nats"]
        point["excess_nats"] = None if surprisal is None else surprisal - entropy


def metric_values(points: list[dict[str, object]], metric: str) -> list[float]:
    key = f"{metric}_nats"
    return [np.nan if point[key] is None else point[key] for point in points]


def metric_matrix(passages: list[dict[str, object]], metric: str) -> np.ndarray:
    return np.array([metric_values(passage["points"], metric) for passage in passages])


def summarize_group(passages: list[dict[str, object]], metric: str) -> dict[str, object]:
    per_passage = [role_means(passage["points"], metric) for passage in passages]
    by_role: dict[str, dict[str, dict[str, float]]] = {}
    for half in per_passage[0]:
        by_role[half] = {}
        for role in per_passage[0][half]:
            values = np.array([means[half][role] for means in per_passage])
            by_role[half][role] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "median": float(np.median(values)),
            }
    matrix = metric_matrix(passages, metric)
    # Position 1 is never scored, so its column is all NaN.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    return {
        "n_passages": len(passages),
        "by_role_across_passages": by_role,
        "per_position": {
            "mean": [None if np.isnan(v) else float(v) for v in np.nanmean(matrix, axis=0)],
            "median": [None if np.isnan(v) else float(v) for v in np.nanmedian(matrix, axis=0)],
            "sd": [None if np.isnan(v) else float(v) for v in np.nanstd(matrix, axis=0, ddof=1)],
        },
    }


def highlight_phrases(axis) -> None:
    for start, end in HIGHLIGHT_RANGES:
        axis.axvspan(start - 0.5, end + 0.5, color=HIGHLIGHT_COLOR, alpha=0.45, linewidth=0, zorder=0)


def draw_group(
    axis,
    passages: list[dict[str, object]],
    example_index: int,
    metric: str,
) -> None:
    matrix = metric_matrix(passages, metric)
    positions = np.arange(1, matrix.shape[1] + 1)
    rng = np.random.default_rng(0)
    for row in matrix:
        jitter = rng.uniform(-0.18, 0.18, size=len(row))
        axis.scatter(positions + jitter, row, s=12, color=COLOR, alpha=0.22, linewidths=0, zorder=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(matrix, axis=0)
    axis.plot(
        positions,
        mean,
        color=COLOR,
        linewidth=2.8,
        marker="o",
        markersize=5,
        zorder=3,
        label="Mean",
    )
    axis.scatter([], [], s=12, color=COLOR, alpha=0.4, label=f"Individual passages (n={len(passages)})")
    style_surprisal_axis(axis, metric)
    example = passages[example_index]
    set_position_axes(axis, list(positions), [point["token"] for point in example["points"]])
    axis.set_xlabel(
        f"Characters of one example passage ({example['id']}); points show all {len(passages)}",
        fontsize=11,
        color="#555555",
    )
    axis.legend(loc="upper right", frameon=False, fontsize=11)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=METRICS, default="surprisal")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--reference-label", default=DEFAULT_REFERENCE_LABEL)
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--group-labels", nargs="+", default=DEFAULT_GROUP_LABELS)
    parser.add_argument("--output-png", default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--example-index",
        type=int,
        default=0,
        help="Which passage of each group supplies the x-axis character labels.",
    )
    parser.add_argument(
        "--shared-y",
        action="store_true",
        help="Give all panels of the stacked figure the same y-axis range.",
    )
    args = parser.parse_args()
    for name in ("output_png", "output_json"):
        setattr(args, name, getattr(args, name).format(metric=args.metric))
    if len(args.groups) != len(args.group_labels):
        raise ValueError("Provide one label per group.")

    load_fonts()
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    groups = [
        json.loads(Path(path).read_text(encoding="utf-8"))["passages"] for path in args.groups
    ]
    add_excess(reference["points"])
    for passages in groups:
        for passage in passages:
            add_excess(passage["points"])
    n_tokens = len(reference["points"])
    for path, passages in zip(args.groups, groups):
        if any(len(passage["points"]) != n_tokens for passage in passages):
            raise ValueError(f"{path}: passages must match the reference length ({n_tokens}).")

    n_panels = 1 + len(groups)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(max(10, min(24, n_tokens * 0.24)), 4.6 * n_panels),
        sharey=args.shared_y,
    )
    draw_surprisal(axes[0], reference["points"], args.metric)
    axes[0].set_title(args.reference_label, fontsize=14, loc="left", pad=26)
    for axis, passages, label in zip(axes[1:], groups, args.group_labels):
        draw_group(axis, passages, args.example_index, args.metric)
        axis.set_title(label, fontsize=14, loc="left", pad=26)
    for axis in axes:
        highlight_phrases(axis)
        if args.metric == "excess":
            axis.axhline(0, color="#555555", linewidth=1, zorder=1)
    fig.tight_layout(h_pad=2.5)

    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "metric": args.metric,
        "content_positions": sorted(MENGZI_CONTENT_POSITIONS),
        "half_boundary_position": MENGZI_HALF_BOUNDARY,
        "reference": {
            "label": args.reference_label,
            "input": args.reference,
            "text": reference["text"],
            f"mean_{args.metric}_nats": float(np.nanmean(metric_values(reference["points"], args.metric))),
            "by_role": role_means(reference["points"], args.metric),
        },
        "groups": [
            {"label": label, "input": path, **summarize_group(passages, args.metric)}
            for path, label, passages in zip(args.groups, args.group_labels, groups)
        ],
    }
    write_json(Path(args.output_json), payload)
    print(f"Saved figure: {output_png}")
    print(f"Saved summary: {args.output_json}")


if __name__ == "__main__":
    main()
