"""
Experiment: compare GujiGPT perplexity under original vs replacement context.

Input is a JSON file with:

    {
      "left_context": "...",  // optional
      "left_part": "...",
      "right_part": "..."
    }

The model is scored on the full text:

    left_context + left_part + right_part

Replacement runs replace the entire conditioning prefix:

    left_context + left_part

with either same-length spans from --replacement-source or a shuffled version of
that same prefix, then append the original right_part. The plot shows only:

    left_part + right_part

so the left side is the end of the conditioning region and the right side is the
shared continuation whose perplexity is affected by the preceding context.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import opencc
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import compact_text, select_device
from lm_scoring_utils import (
    DEFAULT_INPUT_PATH as DEFAULT_REPLACEMENT_SOURCE,
    DEFAULT_MODEL,
    PUNCTUATION,
    causal_lm_context_limit,
    score_char_perplexities,
)

from qhchina import load_fonts


DEFAULT_INPUT_JSON = "data/1_mengzi_prose/context_replacement_prose1.json"
DEFAULT_OUTPUT_PNG = "results/prose_context_replacement/gujigpt_context_replacement_perplexity.png"
DEFAULT_OUTPUT_JSON = "results/prose_context_replacement/gujigpt_context_replacement_perplexity.json"
REPLACEMENT_MODES = ("exact-length", "shuffle")


def load_input_parts(path: Path) -> dict[str, str]:
    """Load the three input text parts, assuming they are already Traditional Chinese."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    left_context = compact_text(payload.get("left_context", ""))
    left_part = compact_text(payload["left_part"])
    right_part = compact_text(payload["right_part"])
    if not left_part:
        raise ValueError("Input JSON must contain a non-empty left_part.")
    if not right_part:
        raise ValueError("Input JSON must contain a non-empty right_part.")
    return {
        "left_context": left_context,
        "left_part": left_part,
        "right_part": right_part,
    }


def read_replacement_source(path: Path, converter: opencc.OpenCC) -> str:
    """Read replacement source as one compact Traditional Chinese line.

    Very short source lines are discarded before whitespace removal so headings,
    fragments, and other metadata do not become replacement material.
    """
    with path.open("r", encoding="utf-8") as f:
        lines = [
            compact_text(converter.convert(line))
            for line in f
        ]
    return "".join(line for line in lines if len(line) >= 8)


def terminal_punctuation(text: str) -> str:
    """Return the punctuation mark that replacement spans must end with."""
    if not text or text[-1] not in PUNCTUATION:
        raise ValueError(
            "The replacement target (left_context + left_part) must end with "
            "a punctuation mark so same-boundary replacements can be sampled."
        )
    return text[-1]


def find_replacement_spans(
    source_text: str,
    target_length: int,
    avoid_text: str,
    count: int,
) -> list[dict[str, Any]]:
    """Find same-length replacement spans ending with the target punctuation."""
    required_terminal = terminal_punctuation(avoid_text)
    preferred: list[tuple[int, int, int, str]] = []
    seen = set()

    for start in range(0, len(source_text) - target_length + 1):
        end = start + target_length
        span = source_text[start:end]
        if span == avoid_text or span in seen:
            continue
        if not span.endswith(required_terminal):
            continue
        seen.add(span)
        starts_cleanly = start == 0 or source_text[start - 1] in PUNCTUATION
        score = 0 if starts_cleanly else 1
        item = (score, start, end, span)
        preferred.append(item)

    candidates = sorted(preferred, key=lambda item: (item[0], item[1]))
    if not candidates:
        raise ValueError(
            "Could not find any same-length replacement span ending with "
            f"{required_terminal!r}."
        )

    replacements = []
    for score, start, end, span in candidates[:count]:
        reason = (
            "punctuation_bounded_and_matching_terminal_punctuation"
            if score == 0
            else "matching_terminal_punctuation"
        )
        replacements.append(
            {
                "text": span,
                "start": start,
                "end": end,
                "reason": reason,
            }
        )
    return replacements


def shuffled_contexts(text: str, count: int) -> list[dict[str, Any]]:
    """Return random permutations of the provided conditioning context."""
    replacements = []
    for sample_index in range(count):
        chars = list(text)
        random.shuffle(chars)
        shuffled = "".join(chars)
        replacements.append(
            {
                "text": shuffled,
                "sample": sample_index + 1,
                "reason": "shuffled_conditioning_context",
            }
        )
    return replacements


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def score_text(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_context_tokens: int,
    score_block_tokens: int,
) -> tuple[list[float | None], list[float | None]]:
    """Return per-character perplexities and losses for a full scored text."""
    scores = score_char_perplexities(
        model=model,
        tokenizer=tokenizer,
        text=text,
        device=device,
        max_context_tokens=max_context_tokens,
        score_block_tokens=score_block_tokens,
    )
    perplexities = [score.perplexity for score in scores]
    losses = [
        -score.log_probability if score.log_probability is not None else None
        for score in scores
    ]
    return perplexities, losses


def finite_mean(values: list[float | None]) -> float | None:
    """Mean over finite numeric values."""
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def average_losses(loss_sets: list[list[float | None]]) -> list[float | None]:
    """Average per-character losses across several replacement runs."""
    if not loss_sets:
        return []
    averaged = []
    for position_losses in zip(*loss_sets, strict=True):
        finite = [
            loss
            for loss in position_losses
            if loss is not None and np.isfinite(loss)
        ]
        averaged.append(float(np.mean(finite)) if finite else None)
    return averaged


def losses_to_perplexities(losses: list[float | None]) -> list[float | None]:
    """Convert average negative log probabilities back to perplexity."""
    perplexities = []
    for loss in losses:
        if loss is None or not np.isfinite(loss):
            perplexities.append(None)
        elif loss < 700:
            perplexities.append(float(math.exp(loss)))
        else:
            perplexities.append(math.inf)
    return perplexities


def point_payload(
    index: int,
    plot_text: str,
    original_perplexities: list[float | None],
    replacement_perplexities: list[float | None],
    original_losses: list[float | None],
    replacement_losses: list[float | None],
) -> dict[str, Any]:
    """Return one plotted point for JSON output."""
    return {
        "position": index + 1,
        "char": plot_text[index],
        "original_perplexity": original_perplexities[index],
        "replacement_perplexity": replacement_perplexities[index],
        "original_loss": original_losses[index],
        "replacement_loss": replacement_losses[index],
    }


def to_plot_array(values: list[float | None]) -> np.ndarray:
    """Convert optional positive values to a NaN-padded plot array."""
    return np.array(
        [
            np.nan if value is None or not np.isfinite(value) or value <= 0 else value
            for value in values
        ],
        dtype=float,
    )


def plot_perplexities(
    plot_text: str,
    original_perplexities: list[float | None],
    replacement_perplexities: list[float | None],
    left_part_length: int,
    full_position_offset: int,
    replacement_count: int,
    output_path: Path,
    y_cap: float | None,
    dpi: int,
) -> None:
    """Write a log-scale character perplexity plot for left_part + right_part."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(1, len(plot_text) + 1)
    original_y = to_plot_array(original_perplexities)
    replacement_y = to_plot_array(replacement_perplexities)
    if y_cap is not None:
        original_y = np.minimum(original_y, y_cap)
        replacement_y = np.minimum(replacement_y, y_cap)

    width = max(14, min(34, len(plot_text) * 0.30))
    fig, ax = plt.subplots(figsize=(width, 5.8))
    original_color = "#d62728"
    replacement_color = "#00008b"
    left_slice = slice(0, left_part_length)
    right_slice = slice(left_part_length, None)

    ax.plot(
        x,
        original_y,
        marker="o",
        markersize=4.4,
        linewidth=3.0,
        color=original_color,
        alpha=0.92,
        label="Original context",
        zorder=4,
    )
    ax.plot(
        x[left_slice],
        replacement_y[left_slice],
        marker="o",
        markersize=3.6,
        linewidth=2.2,
        color=replacement_color,
        alpha=0.10,
        label="_nolegend_",
        zorder=2,
    )
    ax.plot(
        x[right_slice],
        replacement_y[right_slice],
        marker="o",
        markersize=4.4,
        linewidth=3.0,
        color=replacement_color,
        alpha=0.82,
        label=(
            "Replacement context"
            if replacement_count == 1
            else f"Replacement contexts mean (n={replacement_count})"
        ),
        zorder=3,
    )
    if left_part_length < len(plot_text):
        boundary_slice = slice(left_part_length - 1, left_part_length + 1)
        ax.plot(
            x[boundary_slice],
            replacement_y[boundary_slice],
            linewidth=2.2,
            color=replacement_color,
            alpha=0.10,
            label="_nolegend_",
            zorder=3.5,
        )

    ax.axvline(left_part_length + 0.5, color="#444444", linestyle="--", linewidth=1)

    ax.set_yscale("log")
    ax.set_xlim(0.5, len(plot_text) + 0.5)
    ax.set_xlabel("Character")
    ax.set_ylabel("Perplexity (log scale)")
    ax.grid(True, which="both", axis="both", alpha=0.25)
    ax.tick_params(axis="x", labelsize=10, pad=2)
    ax.tick_params(axis="y", labelsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(list(plot_text), rotation=0, fontsize=10)
    ax.text(
        -0.015,
        -0.015,
        "text",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#111111",
        clip_on=False,
    )

    top = ax.secondary_xaxis("top")
    top.set_xticks(x)
    top.set_xticklabels([str(full_position_offset + pos) for pos in x], fontsize=6)
    top.tick_params(length=2)

    ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.subplots_adjust(bottom=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot GujiGPT perplexity for left_part + right_part under original "
            "vs replacement left-context conditioning."
        )
    )
    parser.add_argument("--input-json", default=DEFAULT_INPUT_JSON)
    parser.add_argument(
        "--replacement-source",
        default=DEFAULT_REPLACEMENT_SOURCE,
        help="Text file used as the pool for same-length replacement contexts.",
    )
    parser.add_argument(
        "--replacement-mode",
        choices=REPLACEMENT_MODES,
        default="exact-length",
        help=(
            "How to build replacement contexts: exact-length samples same-length "
            "spans from --replacement-source; shuffle randomly permutes "
            "left_context + left_part."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face causal LM.")
    parser.add_argument("--output-png", default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, etc.")
    parser.add_argument("--score-block-tokens", type=int, default=256)
    parser.add_argument(
        "--replacement-samples",
        type=int,
        default=1,
        help=(
            "Number of replacement contexts to score. When greater than 1, the "
            "replacement curve plots exp(mean loss) at each position."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--y-cap",
        type=float,
        default=None,
        help="Optional cap for plotted perplexities.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_fonts()
    if args.replacement_samples < 1:
        raise ValueError("--replacement-samples must be at least 1")

    input_json = Path(args.input_json)
    replacement_source = Path(args.replacement_source)
    output_png = Path(args.output_png)
    output_json = Path(args.output_json)
    converter = opencc.OpenCC("s2t")

    parts = load_input_parts(input_json)
    left_context = parts["left_context"]
    left_part = parts["left_part"]
    right_part = parts["right_part"]
    conditioning_text = left_context + left_part
    original_full_text = conditioning_text + right_part
    plot_start = len(left_context)
    plot_text = left_part + right_part

    if args.replacement_mode == "exact-length":
        source_text = read_replacement_source(replacement_source, converter)
        replacements = find_replacement_spans(
            source_text=source_text,
            target_length=len(conditioning_text),
            avoid_text=conditioning_text,
            count=args.replacement_samples,
        )
    else:
        replacements = shuffled_contexts(
            text=conditioning_text,
            count=args.replacement_samples,
        )

    device = select_device(args.device)
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Loading model: {args.model} on {device}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    max_context_tokens = causal_lm_context_limit(model, tokenizer)
    print(f"Using context limit: {max_context_tokens} tokens")
    original_full_perplexities, original_full_losses = score_text(
        model=model,
        tokenizer=tokenizer,
        text=original_full_text,
        device=device,
        max_context_tokens=max_context_tokens,
        score_block_tokens=args.score_block_tokens,
    )
    original_perplexities = original_full_perplexities[plot_start:]
    original_losses = original_full_losses[plot_start:]

    replacement_full_loss_sets = []
    for replacement_info in tqdm(replacements, desc="Scoring replacements", unit="sample"):
        replacement_full_text = replacement_info["text"] + right_part
        _replacement_full_perplexities, replacement_full_losses = score_text(
            model=model,
            tokenizer=tokenizer,
            text=replacement_full_text,
            device=device,
            max_context_tokens=max_context_tokens,
            score_block_tokens=args.score_block_tokens,
        )
        replacement_full_loss_sets.append(replacement_full_losses[plot_start:])

    replacement_losses = average_losses(replacement_full_loss_sets)
    replacement_perplexities = losses_to_perplexities(replacement_losses)

    plot_perplexities(
        plot_text=plot_text,
        original_perplexities=original_perplexities,
        replacement_perplexities=replacement_perplexities,
        left_part_length=len(left_part),
        full_position_offset=len(left_context),
        replacement_count=len(replacements),
        output_path=output_png,
        y_cap=args.y_cap,
        dpi=args.dpi,
    )

    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_json": str(input_json),
            "replacement_mode": args.replacement_mode,
            "replacement_source": (
                str(replacement_source)
                if args.replacement_mode == "exact-length"
                else None
            ),
            "model": args.model,
            "device": str(device),
            "input_json_assumed_traditional": True,
            "replacement_opencc_config": "s2t",
            "left_context_length": len(left_context),
            "left_part_length": len(left_part),
            "right_part_length": len(right_part),
            "conditioning_length": len(conditioning_text),
            "plot_text_length": len(plot_text),
            "replacement_samples_requested": args.replacement_samples,
            "replacement_samples_used": len(replacements),
            "score_block_tokens": args.score_block_tokens,
            "max_context_tokens": max_context_tokens,
            "output_png": str(output_png),
        },
        "left_context": left_context,
        "left_part": left_part,
        "right_part": right_part,
        "plot_text": plot_text,
        "replacements": replacements,
        "original_mean_perplexity": finite_mean(original_perplexities),
        "replacement_mean_perplexity": finite_mean(replacement_perplexities),
        "original_mean_loss": finite_mean(original_losses),
        "replacement_mean_loss": finite_mean(replacement_losses),
        "points": [
            point_payload(
                index=index,
                plot_text=plot_text,
                original_perplexities=original_perplexities,
                replacement_perplexities=replacement_perplexities,
                original_losses=original_losses,
                replacement_losses=replacement_losses,
            )
            for index in range(len(plot_text))
        ],
    }
    write_json(output_json, payload)
    print(f"Saved plot: {output_png}")
    print(f"Saved data: {output_json}")
    print(f"Replacement samples used: {len(replacements)}")


if __name__ == "__main__":
    main()
