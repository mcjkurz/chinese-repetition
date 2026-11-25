"""Compare mean per-character surprisal or entropy per phrase across passages.

The reference text is split into phrases after each 「，」「；」「。」 (the
punctuation mark belongs to the phrase before it). Control passages share the
template, so the same position ranges are used for every passage. For each
passage the chosen --metric is averaged over the scored characters of each
phrase. The stacked figure shows the reference and each control group (faint
per-passage points, mean on top). Output file names end in _surprisal or
_entropy.
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
from sequence_surprisal import style_surprisal_axis, write_json
from surprisal_comparison import (
    COLOR,
    DEFAULT_GROUP_LABELS,
    DEFAULT_GROUPS,
    DEFAULT_REFERENCE,
    DEFAULT_REFERENCE_LABEL,
    METRICS,
    add_excess,
    metric_matrix,
)


DEFAULT_OUTPUT_PNG = "results/surprisal/mengzi_phrases_comparison_{metric}.png"
DEFAULT_OUTPUT_JSON = "results/surprisal/mengzi_phrases_summary_{metric}.json"
PHRASE_BREAKS = set("，；。")


def split_phrases(text: str) -> list[tuple[int, int]]:
    """Return 1-based inclusive (start, end) positions of each phrase."""
    phrases = []
    start = 1
    for position, character in enumerate(text, start=1):
        if character in PHRASE_BREAKS:
            phrases.append((start, position))
            start = position + 1
    if start <= len(text):
        phrases.append((start, len(text)))
    return phrases


def phrase_matrix(
    passages: list[dict[str, object]],
    phrases: list[tuple[int, int]],
    metric: str,
) -> np.ndarray:
    """Rows are passages, columns are phrases; values are per-character means."""
    matrix = metric_matrix(passages, metric)
    with warnings.catch_warnings():
        # Position 1 is never scored, so a one-character phrase there is all NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.column_stack(
            [np.nanmean(matrix[:, start - 1 : end], axis=1) for start, end in phrases]
        )


def phrase_texts(passage: dict[str, object], phrases: list[tuple[int, int]]) -> list[str]:
    tokens = [point["token"] for point in passage["points"]]
    return ["".join(tokens[start - 1 : end]) for start, end in phrases]


def set_phrase_axes(axis, phrases: list[tuple[int, int]], labels: list[str]) -> None:
    """Label phrases below and their position ranges above the x-axis."""
    ticks = np.arange(1, len(phrases) + 1)
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels, fontsize=11)
    axis.tick_params(axis="x", pad=5)
    axis.set_xlim(0.5, len(phrases) + 0.5)

    top = axis.secondary_xaxis("top")
    top.set_xticks(ticks)
    top.set_xticklabels([f"{start}–{end}" for start, end in phrases], fontsize=11)
    top.tick_params(length=2)


def draw_reference(axis, values: np.ndarray, phrases, labels) -> None:
    axis.plot(
        np.arange(1, len(values) + 1),
        values,
        color=COLOR,
        linewidth=2.8,
        marker="o",
        markersize=6,
    )
    set_phrase_axes(axis, phrases, labels)


def draw_group(axis, values: np.ndarray, phrases, labels, example_id: str) -> None:
    ticks = np.arange(1, values.shape[1] + 1)
    rng = np.random.default_rng(0)
    for row in values:
        jitter = rng.uniform(-0.12, 0.12, size=len(row))
        axis.scatter(ticks + jitter, row, s=16, color=COLOR, alpha=0.25, linewidths=0, zorder=2)
    axis.plot(
        ticks,
        values.mean(axis=0),
        color=COLOR,
        linewidth=2.8,
        marker="o",
        markersize=6,
        zorder=3,
        label="Mean",
    )
    axis.scatter([], [], s=16, color=COLOR, alpha=0.4, label=f"Individual passages (n={len(values)})")
    set_phrase_axes(axis, phrases, labels)
    axis.set_xlabel(
        f"Phrases of one example passage ({example_id}); points show all {len(values)}",
        fontsize=11,
        color="#555555",
    )
    axis.legend(loc="upper right", frameon=False, fontsize=11)


def describe(values: np.ndarray) -> dict[str, list[float | None]]:
    def clean(row):
        return [None if np.isnan(v) else float(v) for v in row]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {
            "mean": clean(np.nanmean(values, axis=0)),
            "sd": clean(np.nanstd(values, axis=0, ddof=1)),
            "median": clean(np.nanmedian(values, axis=0)),
        }


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
        help="Which passage of each group supplies the x-axis phrase labels.",
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

    phrases = split_phrases(reference["text"])
    reference_values = phrase_matrix([reference], phrases, args.metric)[0]
    reference_labels = phrase_texts(reference, phrases)
    group_values = [phrase_matrix(passages, phrases, args.metric) for passages in groups]

    n_panels = 1 + len(groups)
    fig, axes = plt.subplots(n_panels, 1, figsize=(18, 4.6 * n_panels), sharey=args.shared_y)
    draw_reference(axes[0], reference_values, phrases, reference_labels)
    style_surprisal_axis(axes[0], args.metric)
    axes[0].set_title(args.reference_label, fontsize=14, loc="left", pad=26)
    for axis, passages, values, label in zip(axes[1:], groups, group_values, args.group_labels):
        example = passages[args.example_index]
        draw_group(axis, values, phrases, phrase_texts(example, phrases), example["id"])
        style_surprisal_axis(axis, args.metric)
        axis.set_title(label, fontsize=14, loc="left", pad=26)
    for axis in axes:
        axis.set_ylabel(f"Mean {args.metric} per character (nats)", fontsize=12)
    fig.tight_layout(h_pad=2.5)
    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "metric": args.metric,
        "value": f"mean per-character {args.metric} (nats) over the scored characters of each phrase",
        "phrases": [
            {"start": start, "end": end, "reference_text": text}
            for (start, end), text in zip(phrases, reference_labels)
        ],
        "reference": {
            "label": args.reference_label,
            "input": args.reference,
            "values": [None if np.isnan(v) else float(v) for v in reference_values],
        },
        "groups": [
            {"label": label, "input": path, "n_passages": len(values), **describe(values)}
            for path, label, values in zip(args.groups, args.group_labels, group_values)
        ],
    }
    write_json(Path(args.output_json), payload)
    print(f"Saved figure: {output_png}")
    print(f"Saved summary: {args.output_json}")


if __name__ == "__main__":
    main()
