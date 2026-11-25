"""Score and plot next-token surprisal for a supplied text sequence.

Each point also records the entropy of the next-token distribution, so later
plots can show either metric.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import select_device
from lm_scoring_utils import (
    DEFAULT_MODEL,
    causal_lm_context_limit,
    score_token_log_probabilities_and_entropies,
)
from qhchina import load_fonts


DEFAULT_INPUT = "data/1_mengzi_prose/mengzi_surprisal.txt"
DEFAULT_OUTPUT_JSON = "results/surprisal/mengzi_surprisal.json"
DEFAULT_OUTPUT_PNG = "results/surprisal/mengzi_surprisal.png"
METRIC_LABELS = {
    "surprisal": "Surprisal (nats)",
    "entropy": "Entropy (nats)",
    "excess": "Surprisal − entropy (nats)",
}


def load_sequence(path: Path) -> str:
    if path.suffix.lower() != ".txt":
        raise ValueError(f"Input file must be a .txt file: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No text sequence found in {path}.")
    return text


def score_surprisal(model, tokenizer, text: str, device, max_context_tokens: int):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    input_ids = encoded["input_ids"].to(device)
    if input_ids.shape[1] == 0:
        raise ValueError("The tokenizer produced no tokens for the input sequence.")

    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    log_probabilities, entropies = score_token_log_probabilities_and_entropies(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_context_tokens=max_context_tokens,
        score_block_tokens=256,
    )
    points = []
    for position, ((start, end), log_probability, entropy) in enumerate(
        zip(offsets, log_probabilities, entropies, strict=True), start=1
    ):
        points.append(
            {
                "position": position,
                "token": text[start:end],
                "surprisal_nats": (
                    None if log_probability is None else -log_probability
                ),
                "entropy_nats": entropy,
            }
        )
    return points


def set_position_axes(
    axis,
    positions: list[int],
    tokens: list[str],
    token_color: str = "black",
) -> None:
    """Label tokens below and 1-based positions above the x-axis."""
    step = max(1, len(positions) // 80)
    selected_positions = positions[::step]
    axis.set_xticks(selected_positions)
    axis.set_xticklabels(tokens[::step], fontsize=12, color=token_color)
    axis.tick_params(axis="x", labelsize=12, pad=5)
    axis.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)

    top = axis.secondary_xaxis("top")
    top.set_xticks(selected_positions)
    top.set_xticklabels([str(position) for position in selected_positions], fontsize=12)
    top.tick_params(length=2)


def style_surprisal_axis(axis, metric: str = "surprisal") -> None:
    axis.set_ylabel(METRIC_LABELS[metric], fontsize=12)
    axis.set_axisbelow(True)
    axis.grid(
        visible=True,
        axis="both",
        color="#7f8c8d",
        linewidth=0.7,
        alpha=0.28,
    )


def draw_surprisal(
    axis,
    points: list[dict[str, object]],
    metric: str = "surprisal",
) -> None:
    """Draw a surprisal (or entropy) curve with tokens below and positions above."""
    key = f"{metric}_nats"
    positions = [point["position"] for point in points]
    scores = [np.nan if point[key] is None else point[key] for point in points]
    axis.plot(
        positions,
        scores,
        color="#2878a5",
        linewidth=2.8,
        marker="o",
        markersize=4.5,
    )
    style_surprisal_axis(axis, metric)
    set_position_axes(axis, positions, [point["token"] for point in points])


def plot_surprisal(
    path: Path,
    points: list[dict[str, object]],
    title: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(max(10, min(24, len(points) * 0.24)), 5.5))
    draw_surprisal(axis, points)
    if title:
        axis.set_title(title, fontsize=15, pad=12)
    axis.set_xlabel("Token position", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score and plot next-token surprisal for a text sequence."
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", help="Path to a UTF-8 .txt file.")
    inputs.add_argument("--text", help="Text sequence to score directly.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-png", default=DEFAULT_OUTPUT_PNG)
    parser.add_argument(
        "--title",
        help="Optional figure title. No title is shown if this is omitted.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    load_fonts()
    if args.text is not None:
        text = args.text
        input_source = "command line text"
    else:
        input_path = Path(args.input or DEFAULT_INPUT)
        text = load_sequence(input_path)
        input_source = str(input_path)

    if not text:
        raise ValueError("The input sequence is empty.")

    device = select_device(args.device)
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Loading model: {args.model} on {device}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    points = score_surprisal(
        model=model,
        tokenizer=tokenizer,
        text=text,
        device=device,
        max_context_tokens=causal_lm_context_limit(model, tokenizer),
    )
    output_png = Path(args.output_png)
    plot_surprisal(output_png, points, title=args.title)
    observed = [
        point["surprisal_nats"]
        for point in points
        if point["surprisal_nats"] is not None
    ]
    payload: dict[str, object] = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": input_source,
            "model": args.model,
            "device": str(device),
            "metric": "next-token surprisal in nats: -ln p(x_t | x_<t)",
            "entropy": "entropy in nats of the next-token distribution p(. | x_<t)",
            "first_token": "unscored because no preceding token is provided",
            "output_png": str(output_png),
            "title": args.title,
        },
        "text": text,
        "mean_surprisal_nats": (
            float(np.mean(observed)) if observed else None
        ),
        "mean_entropy_nats": float(
            np.mean([p["entropy_nats"] for p in points if p["entropy_nats"] is not None])
        ),
        "points": points,
    }
    write_json(Path(args.output_json), payload)
    print(f"Saved figure: {output_png}")
    print(f"Saved surprisal data: {args.output_json}")


if __name__ == "__main__":
    main()
