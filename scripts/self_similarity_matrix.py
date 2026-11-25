"""Compute and plot a character-level self-similarity matrix for an encoder model."""

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
from transformers import AutoModel, AutoTokenizer
from qhchina import load_fonts

from common import compact_text, select_device

DEFAULT_MODEL = "hsc748NLP/GujiBERT_fan"
DEFAULT_CMAP = "viridis"
LEG_OPEN_TAG = "<leg>"
LEG_CLOSE_TAG = "</leg>"


def parse_leg_marked_text(text: str) -> tuple[str, list[tuple[int, int]], list[str]]:
    compact = compact_text(text)
    output: list[str] = []
    leg_spans: list[tuple[int, int]] = []
    leg_texts: list[str] = []
    in_leg = False
    leg_start: int | None = None
    cursor = 0

    while cursor < len(compact):
        if compact.startswith(LEG_OPEN_TAG, cursor):
            if in_leg:
                raise ValueError("Nested <leg> tags are not supported")
            in_leg = True
            leg_start = len(output)
            cursor += len(LEG_OPEN_TAG)
        elif compact.startswith(LEG_CLOSE_TAG, cursor):
            if not in_leg or leg_start is None:
                raise ValueError("Found </leg> without a matching <leg>")
            leg_end = len(output)
            leg_spans.append((leg_start, leg_end))
            leg_texts.append("".join(output[leg_start:leg_end]))
            in_leg = False
            leg_start = None
            cursor += len(LEG_CLOSE_TAG)
        else:
            output.append(compact[cursor])
            cursor += 1

    if in_leg:
        raise ValueError("Found <leg> without a matching </leg>")

    return "".join(output), leg_spans, leg_texts


def scramble_text(text: str, rng: random.Random | None = None) -> str:
    chars = list(text)
    (rng or random).shuffle(chars)
    return "".join(chars)


def normalize_layer_index(layer_index: int, num_hidden_layers: int) -> int:
    layer_count = num_hidden_layers + 1
    if layer_index < 0:
        layer_index = layer_count + layer_index
    if not 0 <= layer_index < layer_count:
        raise ValueError(f"--layer must be in [0, {layer_count - 1}] or a valid negative index")
    return layer_index


def all_layer_character_embeddings(
    text: str,
    model,
    tokenizer,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    """Return visible tokens and their hidden states for every layer, shape (layers, tokens, dim).

    Texts longer than the model's context are encoded in overlapping windows
    (stride of half a window); each token keeps the embedding from the window
    in which it lies closest to the centre.
    """
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    window = int(model.config.max_position_embeddings) - 2
    if len(token_ids) <= window:
        starts = [0]
    else:
        stride = window // 2
        starts = list(range(0, len(token_ids) - window, stride)) + [len(token_ids) - window]

    layer_count = int(model.config.num_hidden_layers) + 1
    hidden = np.zeros((layer_count, len(token_ids), int(model.config.hidden_size)), dtype=np.float32)
    best_distance = np.full(len(token_ids), np.inf)
    for start in starts:
        end = min(start + window, len(token_ids))
        ids = [tokenizer.cls_token_id, *token_ids[start:end], tokenizer.sep_token_id]
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)
        states = torch.stack(outputs.hidden_states)[:, 0, 1:-1].detach().cpu().numpy()
        centre = (start + end - 1) / 2
        for offset, position in enumerate(range(start, end)):
            distance = abs(position - centre)
            if distance < best_distance[position]:
                best_distance[position] = distance
                hidden[:, position] = states[:, offset]

    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    keep = [index for index, token in enumerate(tokens) if token not in {"[CLS]", "[SEP]", "[PAD]"}]
    chars = [tokens[index][2:] if tokens[index].startswith("##") else tokens[index] for index in keep]
    return chars, hidden[:, keep]


def character_embeddings(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    layer_index: int,
) -> tuple[list[str], np.ndarray]:
    selected_layer = normalize_layer_index(layer_index, int(model.config.num_hidden_layers))
    chars, hidden = all_layer_character_embeddings(text, model, tokenizer, device)
    return chars, hidden[selected_layer]


def cosine_similarity_matrix(embeddings: np.ndarray, center: bool = False) -> np.ndarray:
    if center:
        embeddings = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    return normalized @ normalized.T


def summarize_similarity_arrays(arrays: list[np.ndarray]) -> dict[str, float | int | None]:
    count = sum(int(array.size) for array in arrays)
    if count == 0:
        return {"mean": None, "count": 0}
    total = sum(float(np.sum(array)) for array in arrays)
    return {"mean": total / count, "count": count}


def within_block_values(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if len(indices) < 2:
        return np.array([], dtype=np.float32)
    block = matrix[np.ix_(indices, indices)]
    return block[np.triu_indices(len(indices), k=1)]


def cross_block_values(matrix: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.array([], dtype=np.float32)
    return matrix[np.ix_(left, right)].ravel()


def summarize_within_blocks(matrix: np.ndarray, blocks: list[np.ndarray]) -> dict[str, float | int | None]:
    return summarize_similarity_arrays([within_block_values(matrix, block) for block in blocks])


def summarize_between_blocks(matrix: np.ndarray, blocks: list[np.ndarray]) -> dict[str, float | int | None]:
    arrays = [
        cross_block_values(matrix, blocks[left_index], blocks[right_index])
        for left_index in range(len(blocks))
        for right_index in range(left_index + 1, len(blocks))
    ]
    return summarize_similarity_arrays(arrays)


def add_contrast(
    left: dict[str, float | int | None],
    right: dict[str, float | int | None],
    left_name: str,
    right_name: str,
) -> dict[str, float | None]:
    left_mean = left.get("mean")
    right_mean = right.get("mean")
    if not isinstance(left_mean, (int, float)) or not isinstance(right_mean, (int, float)):
        return {f"{left_name}_minus_{right_name}": None, f"{left_name}_over_{right_name}": None}
    ratio = None if abs(float(right_mean)) < 1e-12 else float(left_mean) / float(right_mean)
    return {
        f"{left_name}_minus_{right_name}": float(left_mean) - float(right_mean),
        f"{left_name}_over_{right_name}": ratio,
    }


def leg_similarity_metrics(matrix: np.ndarray, leg_spans: list[tuple[int, int]]) -> dict:
    leg_blocks = [np.arange(start, end) for start, end in leg_spans]
    pair_blocks = [
        np.concatenate([leg_blocks[index], leg_blocks[index + 1]])
        for index in range(0, len(leg_blocks) - 1, 2)
    ]
    adjacent_pair_indexes = {
        (index, index + 1)
        for index in range(0, len(leg_blocks) - 1, 2)
    }

    within_legs = summarize_within_blocks(matrix, leg_blocks)
    between_legs = summarize_between_blocks(matrix, leg_blocks)
    within_pairs = summarize_within_blocks(matrix, pair_blocks)
    between_pairs = summarize_between_blocks(matrix, pair_blocks)
    paired_cross = summarize_similarity_arrays(
        [
            cross_block_values(matrix, leg_blocks[index], leg_blocks[index + 1])
            for index in range(0, len(leg_blocks) - 1, 2)
        ]
    )
    non_paired_cross = summarize_similarity_arrays(
        [
            cross_block_values(matrix, leg_blocks[left_index], leg_blocks[right_index])
            for left_index in range(len(leg_blocks))
            for right_index in range(left_index + 1, len(leg_blocks))
            if (left_index, right_index) not in adjacent_pair_indexes
        ]
    )

    return {
        "leg_count": len(leg_blocks),
        "individual_legs": {
            "within": within_legs,
            "between": between_legs,
            **add_contrast(within_legs, between_legs, "within", "between"),
        },
        "paired_leg_units": {
            "pair_count": len(pair_blocks),
            "ignored_trailing_leg": len(leg_blocks) % 2 == 1,
            "within": within_pairs,
            "between": between_pairs,
            **add_contrast(within_pairs, between_pairs, "within", "between"),
        },
        "paired_vs_non_paired_legs": {
            "paired_cross": paired_cross,
            "non_paired_cross": non_paired_cross,
            **add_contrast(paired_cross, non_paired_cross, "paired_cross", "non_paired_cross"),
        },
    }


def flatten_numeric_metrics(metrics: dict, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in metrics.items():
        metric_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_numeric_metrics(value, metric_key))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if key.endswith("count") or key == "leg_count":
                continue
            flattened[metric_key] = float(value)
    return flattened


def summarize_scrambled_baselines(baselines: list[dict], original_metrics: dict) -> dict:
    if not baselines:
        return {"count": 0, "metrics": {}}

    original_flat = flatten_numeric_metrics(original_metrics)
    flattened_baselines = [flatten_numeric_metrics(baseline) for baseline in baselines]
    metric_keys = sorted({key for baseline in flattened_baselines for key in baseline})
    summaries = {}
    for key in metric_keys:
        values = np.array(
            [baseline[key] for baseline in flattened_baselines if key in baseline],
            dtype=np.float32,
        )
        if values.size == 0:
            continue
        summary = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
        if key in original_flat:
            summary["original"] = original_flat[key]
            summary["proportion_at_least_original"] = float(
                (np.sum(values >= original_flat[key]) + 1) / (values.size + 1)
            )
        summaries[key] = summary

    return {"count": len(baselines), "metrics": summaries}


def compute_scrambled_baselines(
    text: str,
    leg_spans: list[tuple[int, int]],
    baseline_count: int,
    model,
    tokenizer,
    device: torch.device,
    layer_index: int,
    center_embeddings: bool,
    rng: random.Random,
) -> list[dict]:
    baselines = []
    for baseline_index in range(baseline_count):
        scrambled = scramble_text(text, rng)
        chars, embeddings = character_embeddings(scrambled, model, tokenizer, device, layer_index)
        if len(chars) != len(scrambled):
            print(
                f"Warning: skipping scrambled baseline {baseline_index + 1}; "
                f"tokenized {len(chars)} visible tokens from {len(scrambled)} input characters"
            )
            continue
        baselines.append(leg_similarity_metrics(cosine_similarity_matrix(embeddings, center=center_embeddings), leg_spans))
    return baselines


def save_matrix_plot(chars: list[str], matrix: np.ndarray, output_path: Path, dpi: int, cmap: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    size = max(5.5, min(14.0, len(chars) * 0.32))
    fig, ax = plt.subplots(figsize=(size, size))
    finite_values = matrix[np.isfinite(matrix)]
    vmin = float(np.min(finite_values)) if finite_values.size else 0.0
    vmax = float(np.max(finite_values)) if finite_values.size else 1.0
    if abs(vmax - vmin) < 1e-12:
        vmin -= 0.5
        vmax += 0.5

    image = ax.imshow(
        matrix,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.colorbar(image, ax=ax, label="Cosine Similarity", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot character-level encoder self-similarity")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Encoder model name or path")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name or path (default: same as model)")
    parser.add_argument("--text", type=str, default=None, help="Input text to encode")
    parser.add_argument("--input-file", type=str, default=None, help="Text file to encode")
    parser.add_argument(
        "--scramble",
        action="store_true",
        help="Randomly shuffle input characters before computing the matrix",
    )
    parser.add_argument("--layer", type=int, default=-1, help="Encoder layer to use (default: -1, final layer)")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, or mps")
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/prose_similarity/self_similarity_matrix.json",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default="results/prose_similarity/self_similarity_matrix.png",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--cmap", type=str, default=DEFAULT_CMAP, help="Matplotlib colormap for the heatmap")
    parser.add_argument(
        "--center-embeddings",
        action="store_true",
        help="Subtract the mean character embedding before computing cosine similarity",
    )
    parser.add_argument(
        "--baseline-count",
        type=int,
        default=0,
        help="Number of character-scrambled leg-metric baselines to compute",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for scrambling")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_fonts()
    if bool(args.text) == bool(args.input_file):
        raise ValueError("Provide exactly one of --text or --input-file")

    raw_text = args.text if args.text is not None else Path(args.input_file).read_text(encoding="utf-8")
    original_text, leg_spans, leg_texts = parse_leg_marked_text(raw_text)
    rng = random.Random(args.seed)
    text = scramble_text(original_text, rng) if args.scramble else original_text
    if not text:
        raise ValueError("Input text is empty after whitespace removal")
    if args.baseline_count < 0:
        raise ValueError("--baseline-count must be non-negative")

    device = select_device(args.device)
    tokenizer_name = args.tokenizer or args.model
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)
    print(f"Loading model: {args.model} on {device}")
    model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model.to(device)
    model.eval()

    chars, embeddings = character_embeddings(text, model, tokenizer, device, args.layer)
    if len(chars) != len(text):
        print(f"Warning: tokenized {len(chars)} visible tokens from {len(text)} input characters")

    matrix = cosine_similarity_matrix(embeddings, center=args.center_embeddings)
    leg_metrics = None
    baseline_summary = None
    if leg_spans:
        if len(chars) == len(text):
            leg_metrics = leg_similarity_metrics(matrix, leg_spans)
            if args.baseline_count:
                print(f"Computing {args.baseline_count} scrambled baselines")
                baselines = compute_scrambled_baselines(
                    original_text,
                    leg_spans,
                    args.baseline_count,
                    model,
                    tokenizer,
                    device,
                    args.layer,
                    args.center_embeddings,
                    rng,
                )
                baseline_summary = summarize_scrambled_baselines(baselines, leg_metrics)
        else:
            print("Warning: skipping leg metrics because tokenized output does not align to input characters")

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "model": args.model,
                "tokenizer": tokenizer_name,
                "layer": args.layer,
                "scrambled": args.scramble,
                "centered_embeddings": args.center_embeddings,
                "colormap": args.cmap,
                "original_text": original_text,
                "text": text,
                "chars": chars,
                "leg_spans": [{"start": start, "end": end} for start, end in leg_spans],
                "leg_texts": leg_texts,
                "leg_metrics": leg_metrics,
                "scrambled_baselines": baseline_summary,
                "similarity_matrix": matrix.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_matrix_plot(chars, matrix, Path(args.output_png), args.dpi, args.cmap)
    print(f"Saved JSON: {output_json}")
    print(f"Saved plot: {args.output_png}")
    if leg_metrics is not None:
        individual = leg_metrics["individual_legs"]
        paired = leg_metrics["paired_leg_units"]
        print(
            "Leg metrics: "
            f"within/between={individual['within_over_between']}, "
            f"paired-unit within/between={paired['within_over_between']}"
        )


if __name__ == "__main__":
    main()
