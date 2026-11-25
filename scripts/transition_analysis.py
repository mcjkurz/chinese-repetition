"""
Experiment: Transition alignment for regulated-verse couplets, for both
pentasyllabic and heptasyllabic meters.

This script expects precomputed alignment data in
data/2-3_regulated_verse/couplets_penta_alignment.json and data/2-3_regulated_verse/couplets_hepta_alignment.json.
Use scripts/prepare_data_alignment.py to create those files from
data/2-3_regulated_verse/regulated.csv before running this analysis.

For each couplet, BERT-style contextual embeddings are computed for the
Hanzi characters (5 or 7 per line, depending on meter) in a punctuation-aware
couplet context: line1，line2。 Transition alignment is measured in the full
768-dimensional embedding space by default, or after projecting each couplet
through PCA to N dimensions with --pca-dims / --PCA. This is done for every
hidden layer (embedding layer 0 plus transformer layers 1-12). The main
output is a 13 x 4 matrix per meter: mean transition alignment by model
layer and couplet position.

The two inner couplets (Jaw 頷聯, Neck 頸聯 - positions 2 and 3) must match
word for word since the Tang dynasty, while the two outer couplets (Head
首聯, Tail 尾聯 - positions 1 and 4) may only share a theme. The script's
main contrast tests whether inner-couplet lines are more geometrically
aligned - i.e. whether corresponding character-to-character movement
vectors in line 1 and line 2 point in more similar directions - than
outer-couplet lines, layer by layer, for each meter.

Optionally, the script estimates a position-matched random-pair baseline per
layer from transition vectors sampled across unrelated couplets. Each random
comparison keeps the within-line character transition slot fixed, controlling
for layer-wise anisotropy and shared positional-encoding effects.

Use --layer-center-normalize to first subtract the corpus mean vector for each
layer and L2-normalize every contextual character embedding before computing
transition vectors and random-pair baselines.
"""

import argparse
import json
import os
import random
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from common import (
    INNER_POSITIONS,
    METER_NAMES,
    METERS,
    OUTER_POSITIONS,
    format_optional_float,
    select_device,
    split_couplet,
)
from qhchina import load_fonts


POSITIONS = (1, 2, 3, 4)
LINE_SIDES = (0, 1)
POSITION_NAMES = {
    1: ("Head", "首聯"),
    2: ("Jaw", "頷聯"),
    3: ("Neck", "頸聯"),
    4: ("Tail", "尾聯"),
}
POSITION_COLORS = {
    1: "#338833",
    2: "#2244cc",
    3: "#cc7700",
    4: "#cc3333",
}
LAYERS = tuple(range(13))
DEFAULT_DATA_PATH_TEMPLATE = "data/2-3_regulated_verse/couplets_{meter}_alignment.json"
DEFAULT_MODEL = "hsc748NLP/GujiBERT_fan"
DEFAULT_TOKENIZER = None
DEFAULT_MODEL_LABEL = "GujiBERT"
RESULTS_DIR = "results/transition_gujibert_paper"
RESULTS_FILENAME = "transition_alignment_by_layer_results.json"
MATRIX_FILENAME = "transition_alignment_layer_position_matrix.json"
LINE_PLOT_FILENAME = "transition_alignment_by_layer.png"
BASELINE_FILENAME = "transition_alignment_random_pair_baseline.json"
SUBTRACTED_MATRIX_FILENAME = (
    "transition_alignment_layer_position_matrix_baseline_subtracted.json"
)
SUBTRACTED_LINE_PLOT_FILENAME = (
    "transition_alignment_by_layer_baseline_subtracted.png"
)
RATIO_MATRIX_FILENAME = "transition_alignment_layer_position_matrix_baseline_ratio.json"
RATIO_LINE_PLOT_FILENAME = "transition_alignment_by_layer_baseline_ratio.png"
CONTRAST_STATS_FILENAME = "transition_alignment_inner_outer_stats.json"
CONTRAST_PLOT_FILENAME = "transition_alignment_inner_outer_contrast.png"


def transition_slots_for_meter(meter: int) -> tuple[int, ...]:
    """Return the within-line transition slots for a meter (line_length - 1)."""
    return tuple(range(meter - 1))


def meter_filename(filename: str, meter_slug: str) -> str:
    """Insert a meter slug before a filename's extension."""
    stem, ext = os.path.splitext(filename)
    return f"{stem}_{meter_slug}{ext}"


def position_label(position: int, multiline: bool = False) -> str:
    """Return the traditional regulated-verse name for a couplet position."""
    english, chinese = POSITION_NAMES[position]
    separator = "\n" if multiline else " "
    return f"{english}{separator}({chinese})"

def io_role(position: int) -> str:
    """Return 'inner' or 'outer' for a couplet position."""
    if position in INNER_POSITIONS:
        return "inner"
    if position in OUTER_POSITIONS:
        return "outer"
    raise ValueError(f"Unsupported couplet position: {position}")


def projection_description(pca_dims: int | None) -> str:
    """Describe the embedding space used for alignment."""
    if pca_dims is None:
        return "full embedding space"
    return f"{pca_dims}-dimensional PCA space"


def balance_data_by_position(data: list[dict], max_samples: int, seed: int) -> list[dict]:
    """Return an equal number of records for each couplet position."""
    if max_samples % len(POSITIONS) != 0:
        raise ValueError("--max-samples must be divisible by 4 for position-balanced data")

    buckets = {position: [] for position in POSITIONS}
    for item in data:
        position = item.get("position")
        if position in buckets:
            buckets[position].append(item)

    available_counts = {position: len(items) for position, items in buckets.items()}
    target_per_position = min(max_samples // len(POSITIONS), min(available_counts.values()))
    if target_per_position == 0:
        raise RuntimeError(f"Data does not contain all positions: {available_counts}")

    rng = random.Random(seed)
    balanced = []
    for position in POSITIONS:
        items = buckets[position]
        rng.shuffle(items)
        balanced.extend(items[:target_per_position])

    rng.shuffle(balanced)
    print("Position-balanced data:")
    for position in POSITIONS:
        print(f"  position {position}: {target_per_position}")
    if target_per_position * len(POSITIONS) < min(max_samples, len(data)):
        print(
            f"Using {target_per_position * len(POSITIONS)} records "
            f"to keep positions equally represented"
        )

    return balanced


def load_alignment_data(data_path: str) -> list[dict]:
    """Load prepared alignment JSON."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Alignment data not found: {data_path}. "
            "Create it first with scripts/prepare_data_alignment.py."
        )

    print(f"Loading data: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_batches(items: list[dict], batch_size: int):
    """Yield consecutive item batches."""
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def get_character_embeddings_batch(
    texts: list[str],
    target_indices_batch: list[list[int]],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    device: torch.device,
) -> list[np.ndarray]:
    """Return per-layer contextual character embeddings for each text in a batch.

    Each returned array has shape (13, n_chars, hidden_size), where layer 0 is
    the embedding layer and layers 1-12 are transformer layers.
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = [
            layer.detach().cpu().numpy()
            for layer in outputs.hidden_states
        ]

    batch_embeddings = []
    input_ids_cpu = input_ids.detach().cpu().tolist()
    attention_mask_cpu = attention_mask.detach().cpu().tolist()

    for batch_idx, (ids, mask) in enumerate(zip(input_ids_cpu, attention_mask_cpu)):
        tokens = tokenizer.convert_ids_to_tokens(ids)
        target_token_indices = {
            target_idx + 1
            for target_idx in target_indices_batch[batch_idx]
        }
        layer_embeddings = []
        for layer_states in hidden_states:
            embeddings = []
            for token_idx, (token, keep, embedding) in enumerate(
                zip(tokens, mask, layer_states[batch_idx])
            ):
                if not keep or token in {"[CLS]", "[SEP]", "[PAD]"}:
                    continue
                if token_idx in target_token_indices:
                    embeddings.append(embedding)
            layer_embeddings.append(np.array(embeddings))
        batch_embeddings.append(np.array(layer_embeddings))

    return batch_embeddings


def token_count_matches_text_length(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
) -> bool:
    """Return True when the tokenizer maps each input character to one token."""
    tokens = tokenizer.tokenize(text)
    return len(tokens) == len(text)


def couplet_context_and_targets(
    line1: str,
    line2: str,
    include_punctuation: bool,
) -> tuple[str, list[int]]:
    """Return BERT context and Hanzi target positions for one couplet."""
    if not include_punctuation:
        context = line1 + line2
        return context, list(range(len(context)))

    context = f"{line1}，{line2}。"
    line1_targets = list(range(len(line1)))
    line2_start = len(line1) + 1
    line2_targets = [line2_start + idx for idx in range(len(line2))]
    return context, line1_targets + line2_targets


def validate_token_lengths(
    records: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    include_punctuation: bool,
) -> tuple[list[dict], list[str], list[list[int]], int]:
    """Keep records whose BERT context tokenizes one token per character."""
    valid_records = []
    texts = []
    target_indices_batch = []
    skipped = 0

    for record in records:
        text, target_indices = couplet_context_and_targets(
            record["line1"],
            record["line2"],
            include_punctuation,
        )
        if not token_count_matches_text_length(text, tokenizer):
            skipped += 1
            continue
        valid_records.append(record)
        texts.append(text)
        target_indices_batch.append(target_indices)

    return valid_records, texts, target_indices_batch, skipped


def transition_alignment(coords1: np.ndarray, coords2: np.ndarray) -> dict:
    """
    Measure cosine alignment between corresponding line transition vectors.

    For a couplet with `line_length` characters per line, this yields
    `line_length - 1` transition similarities and their mean.
    """
    if len(coords1) != len(coords2) or len(coords1) < 2:
        return {"transition_sims": [], "mean_alignment": 0.0}

    transitions1 = np.diff(coords1, axis=0)
    transitions2 = np.diff(coords2, axis=0)

    transition_sims = []
    for t1, t2 in zip(transitions1, transitions2):
        norm1 = np.linalg.norm(t1)
        norm2 = np.linalg.norm(t2)
        if norm1 < 1e-12 or norm2 < 1e-12:
            sim = 0.0
        else:
            sim = float(np.dot(t1, t2) / (norm1 * norm2))
        transition_sims.append(sim)

    return {
        "transition_sims": transition_sims,
        "mean_alignment": float(np.mean(transition_sims)) if transition_sims else 0.0,
    }


def pca_project_couplet(
    line1_embeddings: np.ndarray,
    line2_embeddings: np.ndarray,
    pca_dims: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project both lines of one couplet into a shared PCA space."""
    if pca_dims is None:
        return line1_embeddings, line2_embeddings

    combined = np.vstack([line1_embeddings, line2_embeddings])
    max_dims = min(combined.shape)
    if pca_dims > max_dims:
        raise ValueError(
            f"--pca-dims/--PCA must be <= {max_dims} for this couplet; "
            f"got {pca_dims}"
        )

    centered = combined - np.mean(combined, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:pca_dims].T
    line1_len = len(line1_embeddings)
    return projected[:line1_len], projected[line1_len:]


def layer_center_normalize_embeddings(
    embeddings_by_layer: np.ndarray,
    layer_centers: np.ndarray,
) -> np.ndarray:
    """Subtract per-layer corpus centers and L2-normalize token embeddings."""
    centered = embeddings_by_layer - layer_centers[:, None, :]
    norms = np.linalg.norm(centered, axis=2, keepdims=True)
    return np.divide(
        centered,
        norms,
        out=np.zeros_like(centered, dtype=np.float64),
        where=norms > 1e-12,
    )


def layer_transition_alignment(
    line1_embeddings_by_layer: np.ndarray,
    line2_embeddings_by_layer: np.ndarray,
    pca_dims: int | None = None,
) -> dict:
    """Compute transition alignment for every layer."""
    layer_mean_alignments = []
    layer_transition_sims = []

    for layer_idx in range(line1_embeddings_by_layer.shape[0]):
        line1_coords, line2_coords = pca_project_couplet(
            line1_embeddings_by_layer[layer_idx],
            line2_embeddings_by_layer[layer_idx],
            pca_dims,
        )
        alignment = transition_alignment(
            line1_coords,
            line2_coords,
        )
        layer_mean_alignments.append(alignment["mean_alignment"])
        layer_transition_sims.append(alignment["transition_sims"])

    return {
        "layer_mean_alignments": layer_mean_alignments,
        "layer_transition_sims": layer_transition_sims,
    }


def init_random_pair_baseline(
    reservoir_size: int,
    hidden_size: int,
    transition_slots: tuple[int, ...],
    seed: int,
) -> dict:
    """Create bounded storage for position-matched unrelated transition alignment."""
    if reservoir_size <= 0:
        raise ValueError("--baseline-reservoir-size must be positive")

    return {
        "vectors": np.zeros(
            (
                len(LAYERS),
                len(LINE_SIDES),
                len(transition_slots),
                reservoir_size,
                hidden_size,
            ),
            dtype=np.float32,
        ),
        "seen": np.zeros(
            (len(LAYERS), len(LINE_SIDES), len(transition_slots)),
            dtype=np.int64,
        ),
        "sum": np.zeros(len(LAYERS), dtype=np.float64),
        "count": np.zeros(len(LAYERS), dtype=np.int64),
        "reservoir_size": reservoir_size,
        "transition_slots": transition_slots,
        "rng": np.random.default_rng(seed),
    }


def update_random_pair_baseline(
    state: dict,
    transition_vectors_by_layer: np.ndarray,
    pairs_per_transition: int,
) -> None:
    """Update layer-wise baseline using same-slot transitions from unrelated lines."""
    if pairs_per_transition <= 0:
        raise ValueError("--baseline-pairs-per-transition must be positive")

    reservoir = state["vectors"]
    seen = state["seen"]
    rng = state["rng"]
    reservoir_size = state["reservoir_size"]
    transition_slots = state["transition_slots"]
    expected_vectors = len(LINE_SIDES) * len(transition_slots)
    if transition_vectors_by_layer.shape[1] != expected_vectors:
        raise ValueError(
            "Position-matched baseline expects exactly "
            f"{expected_vectors} transition vectors per layer; "
            f"got {transition_vectors_by_layer.shape[1]}"
        )

    for layer_idx in LAYERS:
        current_vectors = transition_vectors_by_layer[layer_idx].astype(np.float32)
        current_vectors = current_vectors.reshape(
            len(LINE_SIDES),
            len(transition_slots),
            -1,
        )

        for line_side in LINE_SIDES:
            opposite_side = 1 - line_side
            for slot_idx in range(len(transition_slots)):
                current_vector = current_vectors[line_side, slot_idx]
                available = min(
                    int(seen[layer_idx, opposite_side, slot_idx]),
                    reservoir_size,
                )

                if available:
                    sample_indices = rng.integers(
                        0,
                        available,
                        size=pairs_per_transition,
                    )
                    sampled_vectors = reservoir[
                        layer_idx,
                        opposite_side,
                        slot_idx,
                        sample_indices,
                    ]
                    dots = np.sum(current_vector[None, :] * sampled_vectors, axis=1)
                    current_norm = np.linalg.norm(current_vector)
                    sampled_norms = np.linalg.norm(sampled_vectors, axis=1)
                    denom = current_norm * sampled_norms
                    sims = np.divide(
                        dots,
                        denom,
                        out=np.zeros_like(dots, dtype=np.float64),
                        where=denom > 1e-12,
                    )
                    state["sum"][layer_idx] += float(np.sum(sims))
                    state["count"][layer_idx] += int(sims.size)

        for line_side in LINE_SIDES:
            for slot_idx in range(len(transition_slots)):
                vector = current_vectors[line_side, slot_idx]
                vector_idx = int(seen[layer_idx, line_side, slot_idx])
                seen[layer_idx, line_side, slot_idx] += 1
                if vector_idx < reservoir_size:
                    reservoir[layer_idx, line_side, slot_idx, vector_idx] = vector
                    continue

                replacement_idx = int(rng.integers(0, vector_idx + 1))
                if replacement_idx < reservoir_size:
                    reservoir[
                        layer_idx,
                        line_side,
                        slot_idx,
                        replacement_idx,
                    ] = vector


def random_pair_baseline_values(state: dict) -> list[float]:
    """Return mean same-slot unrelated transition alignment for each layer."""
    counts = state["count"]
    sums = state["sum"]
    return [
        float(sums[layer_idx] / counts[layer_idx])
        if counts[layer_idx] > 0
        else float("nan")
        for layer_idx in LAYERS
    ]


def valid_records_for_batch(
    batch: list[dict],
    meter: int,
) -> tuple[list[dict], int]:
    """Return structurally valid records for one batch and the skipped count."""
    records = []
    skipped = 0

    for item in batch:
        position = item.get("position")
        if position not in POSITIONS:
            skipped += 1
            continue

        line1 = item.get("line1_traditional")
        line2 = item.get("line2_traditional")
        if line1 is None or line2 is None:
            first_line, second_line = split_couplet(item.get("couplet_traditional", ""))
            line1, line2 = first_line, second_line

        if (
            not line1
            or not line2
            or len(line1) != len(line2)
            or len(line1) != meter
        ):
            skipped += 1
            continue

        records.append(
            {
                "couplet": item.get("couplet", f"{line1}，{line2}"),
                "couplet_traditional": item.get("couplet_traditional", f"{line1}，{line2}"),
                "line1": line1,
                "line2": line2,
                "position": position,
            }
        )

    return records, skipped


def estimate_layer_centers(
    data: list[dict],
    meter: int,
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    device: torch.device,
    include_punctuation: bool,
) -> tuple[np.ndarray, int]:
    """Estimate corpus mean contextual character embeddings for each layer."""
    layer_sums = None
    layer_counts = None
    skipped = 0

    for batch in tqdm(iter_batches(data, batch_size), desc="Layer centers"):
        records, batch_skips = valid_records_for_batch(batch, meter)
        skipped += batch_skips
        if not records:
            continue

        records, texts, target_indices_batch, token_skips = validate_token_lengths(
            records,
            tokenizer,
            include_punctuation,
        )
        skipped += token_skips
        if not records:
            continue

        embeddings_batch = get_character_embeddings_batch(
            texts,
            target_indices_batch,
            tokenizer,
            model,
            device,
        )
        for record, embeddings_by_layer in zip(records, embeddings_batch):
            line_len = len(record["line1"])
            if embeddings_by_layer.shape[1] < line_len * 2:
                skipped += 1
                continue

            embeddings_by_layer = embeddings_by_layer[:, :line_len * 2, :]
            if layer_sums is None:
                layer_sums = np.zeros(
                    (embeddings_by_layer.shape[0], embeddings_by_layer.shape[2]),
                    dtype=np.float64,
                )
                layer_counts = np.zeros(embeddings_by_layer.shape[0], dtype=np.int64)

            layer_sums += np.sum(embeddings_by_layer, axis=1)
            layer_counts += embeddings_by_layer.shape[1]

    if layer_sums is None or layer_counts is None or np.any(layer_counts == 0):
        raise RuntimeError("Could not estimate layer centers from the selected data")

    return layer_sums / layer_counts[:, None], skipped


def summarize(values: list[float]) -> str:
    """Format summary statistics for transition alignment values."""
    if not values:
        return "n=0"

    arr = np.array(values)
    return (
        f"n={len(arr):5d}  "
        f"mean={np.mean(arr):.4f}  "
        f"std={np.std(arr):.4f}  "
        f"min={np.min(arr):.4f}  "
        f"max={np.max(arr):.4f}"
    )


def format_p_value(value: float) -> str:
    """Format p-values conservatively for printed reports."""
    if np.isnan(value):
        return "n/a"
    if value < 0.001:
        return "<0.001"
    return f"{value:.6f}"


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original order."""
    adjusted = [np.nan] * len(p_values)
    valid = [(idx, p) for idx, p in enumerate(p_values) if not np.isnan(p)]
    ordered = sorted(valid, key=lambda item: item[1])
    running_max = 0.0
    m = len(ordered)

    for rank, (idx, p_value) in enumerate(ordered):
        corrected = min((m - rank) * p_value, 1.0)
        running_max = max(running_max, corrected)
        adjusted[idx] = running_max

    return adjusted


def bonferroni_adjust(p_values: list[float]) -> list[float]:
    """Return Bonferroni-adjusted p-values in the original order."""
    valid_count = sum(not np.isnan(p_value) for p_value in p_values)
    return [
        min(p_value * valid_count, 1.0) if not np.isnan(p_value) else np.nan
        for p_value in p_values
    ]


def layer_position_values(results: dict, layer_idx: int) -> dict[int, np.ndarray]:
    """Return per-couplet alignment values by position for one layer."""
    values = {}
    for position in POSITIONS:
        values[position] = np.array(
            [
                record["layer_mean_alignments"][layer_idx]
                for record in results[position]
                if len(record.get("layer_mean_alignments", [])) > layer_idx
            ],
            dtype=float,
        )
    return values


def inner_outer_contrast(values_by_position: dict[int, np.ndarray]) -> float:
    """Return mean inner-couplet alignment minus mean outer-couplet alignment."""
    if any(len(values_by_position[position]) == 0 for position in POSITIONS):
        return np.nan

    inner_mean = np.mean([np.mean(values_by_position[position]) for position in INNER_POSITIONS])
    outer_mean = np.mean([np.mean(values_by_position[position]) for position in OUTER_POSITIONS])
    return float(inner_mean - outer_mean)


def independent_cohens_d(values_by_position: dict[int, np.ndarray]) -> float:
    """Return Cohen's d for pooled inner vs outer couplet alignments."""
    inner = np.concatenate([values_by_position[position] for position in INNER_POSITIONS])
    outer = np.concatenate([values_by_position[position] for position in OUTER_POSITIONS])
    if len(inner) < 2 or len(outer) < 2:
        return np.nan

    pooled_var = (
        ((len(inner) - 1) * np.var(inner, ddof=1))
        + ((len(outer) - 1) * np.var(outer, ddof=1))
    ) / (len(inner) + len(outer) - 2)
    if pooled_var <= 0:
        return np.nan
    return float(inner_outer_contrast(values_by_position) / np.sqrt(pooled_var))


def bootstrap_contrast_ci(
    values_by_position: dict[int, np.ndarray],
    rng: np.random.Generator,
    num_bootstrap: int,
    ci_level: float,
    chunk_size: int = 100,
) -> tuple[float, float]:
    """Return a stratified bootstrap CI for the inner-minus-outer contrast."""
    if num_bootstrap <= 0 or any(len(values_by_position[position]) == 0 for position in POSITIONS):
        return np.nan, np.nan

    inner_positions = tuple(sorted(INNER_POSITIONS))
    outer_positions = tuple(sorted(OUTER_POSITIONS))
    bootstrapped = np.empty(num_bootstrap, dtype=float)
    for start in range(0, num_bootstrap, chunk_size):
        stop = min(start + chunk_size, num_bootstrap)
        chunk_len = stop - start
        sampled_means = {}
        for position in POSITIONS:
            values = values_by_position[position]
            sample_indices = rng.integers(
                0,
                len(values),
                size=(chunk_len, len(values)),
            )
            sampled_means[position] = values[sample_indices].mean(axis=1)

        inner_chunk_mean = np.mean(
            [sampled_means[position] for position in inner_positions], axis=0
        )
        outer_chunk_mean = np.mean(
            [sampled_means[position] for position in outer_positions], axis=0
        )
        bootstrapped[start:stop] = inner_chunk_mean - outer_chunk_mean

    alpha = 1.0 - ci_level
    low, high = np.quantile(bootstrapped, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def layer_values_matrix_by_position(results: dict) -> dict[int, np.ndarray]:
    """Return per-couplet layer alignment matrices by position."""
    matrices = {}
    for position in POSITIONS:
        matrices[position] = np.array(
            [
                record["layer_mean_alignments"]
                for record in results[position]
                if len(record.get("layer_mean_alignments", [])) == len(LAYERS)
            ],
            dtype=float,
        )
    return matrices


def permutation_inner_outer_p_values(
    results: dict,
    rng: np.random.Generator,
    num_permutations: int,
    chunk_size: int = 10,
) -> list[float]:
    """Return layer-wise p-values using shared inner/outer permutations."""
    if num_permutations <= 0:
        return [np.nan] * len(LAYERS)

    matrices = layer_values_matrix_by_position(results)
    if any(len(matrices[position]) == 0 for position in POSITIONS):
        return [np.nan] * len(LAYERS)

    inner = np.vstack([matrices[position] for position in sorted(INNER_POSITIONS)])
    outer = np.vstack([matrices[position] for position in sorted(OUTER_POSITIONS)])
    values = np.vstack([inner, outer])
    inner_count = len(inner)
    total_count = len(values)
    observed = np.abs(np.mean(inner, axis=0) - np.mean(outer, axis=0))
    total_sum = np.sum(values, axis=0)
    extreme_counts = np.zeros(len(LAYERS), dtype=np.int64)

    for start in tqdm(
        range(0, num_permutations, chunk_size),
        desc="Inner/outer permutations",
        leave=False,
    ):
        stop = min(start + chunk_size, num_permutations)
        current_chunk_size = stop - start
        random_keys = rng.random((current_chunk_size, total_count))
        inner_indices = np.argpartition(
            random_keys,
            inner_count - 1,
            axis=1,
        )[:, :inner_count]
        inner_sums = values[inner_indices].sum(axis=1)
        outer_sums = total_sum - inner_sums
        permuted = (
            inner_sums / inner_count
            - outer_sums / (total_count - inner_count)
        )
        extreme_counts += np.sum(np.abs(permuted) >= observed, axis=0)

    return [
        float((count + 1.0) / (num_permutations + 1.0))
        for count in extreme_counts
    ]


def compute_inner_outer_stats(
    results: dict,
    num_permutations: int,
    num_bootstrap: int,
    ci_level: float,
    seed: int,
    p_adjust: str,
) -> list[dict]:
    """Compute inner-vs-outer transition alignment tests for each layer."""
    rng = np.random.default_rng(seed)
    rows = []
    p_values = permutation_inner_outer_p_values(
        results,
        rng,
        num_permutations,
    )

    for layer_idx in tqdm(LAYERS, desc="Bootstrap CIs", leave=False):
        values_by_position = layer_position_values(results, layer_idx)
        ci_low, ci_high = bootstrap_contrast_ci(
            values_by_position,
            rng,
            num_bootstrap,
            ci_level,
        )
        position_means = {
            position: (
                float(np.mean(values_by_position[position]))
                if len(values_by_position[position])
                else np.nan
            )
            for position in POSITIONS
        }

        rows.append(
            {
                "layer": layer_idx,
                "n_pos1": int(len(values_by_position[1])),
                "n_pos2": int(len(values_by_position[2])),
                "n_pos3": int(len(values_by_position[3])),
                "n_pos4": int(len(values_by_position[4])),
                "mean_pos1": position_means[1],
                "mean_pos2": position_means[2],
                "mean_pos3": position_means[3],
                "mean_pos4": position_means[4],
                "inner_outer_diff": inner_outer_contrast(values_by_position),
                "cohens_d": independent_cohens_d(values_by_position),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_values[layer_idx],
                "p_adjusted": np.nan,
            }
        )

    p_values = [row["p_value"] for row in rows]
    if p_adjust == "holm":
        adjusted = holm_adjust(p_values)
    elif p_adjust == "bonferroni":
        adjusted = bonferroni_adjust(p_values)
    else:
        adjusted = [np.nan] * len(p_values)

    for row, adjusted_p_value in zip(rows, adjusted):
        row["p_adjusted"] = adjusted_p_value

    return rows


def print_inner_outer_stats(rows: list[dict], ci_level: float, p_adjust: str) -> None:
    """Print inner-vs-outer significance tests by layer."""
    print("\n" + "=" * 118)
    print("INNER VS OUTER TRANSITION ALIGNMENT TESTS")
    print("=" * 118)
    print(
        f"Contrast: mean(position {sorted(INNER_POSITIONS)}) - "
        f"mean(position {sorted(OUTER_POSITIONS)}), using couplets as units."
    )
    print(
        f"CI = {ci_level * 100.0:.1f}% stratified bootstrap CI; "
        "p = two-sided independent-label permutation test."
    )
    if p_adjust != "none":
        print(f"Adjusted p-values are {p_adjust} corrections across the 13 layers.")

    header = (
        f"{'layer':>5}"
        f"{'pos1':>10}"
        f"{'pos2':>10}"
        f"{'pos3':>10}"
        f"{'pos4':>10}"
        f"{'in-out':>12}"
        f"{'d':>10}"
        f"{'ci_low':>12}"
        f"{'ci_high':>12}"
        f"{'p':>12}"
        f"{'p_adj':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['layer']:>5d}"
            f"{format_optional_float(row['mean_pos1']):>10}"
            f"{format_optional_float(row['mean_pos2']):>10}"
            f"{format_optional_float(row['mean_pos3']):>10}"
            f"{format_optional_float(row['mean_pos4']):>10}"
            f"{format_optional_float(row['inner_outer_diff']):>12}"
            f"{format_optional_float(row['cohens_d']):>10}"
            f"{format_optional_float(row['ci_low']):>12}"
            f"{format_optional_float(row['ci_high']):>12}"
            f"{format_p_value(row['p_value']):>12}"
            f"{format_p_value(row['p_adjusted']):>12}"
        )


def build_layer_position_matrix(results: dict) -> np.ndarray:
    """Return a 13 x 4 matrix of mean alignment by layer and couplet position."""
    matrix = np.full((len(LAYERS), len(POSITIONS)), np.nan, dtype=float)
    for layer_idx in LAYERS:
        for position_idx, position in enumerate(POSITIONS):
            values = [
                record["layer_mean_alignments"][layer_idx]
                for record in results[position]
                if len(record.get("layer_mean_alignments", [])) > layer_idx
            ]
            if values:
                matrix[layer_idx, position_idx] = float(np.mean(values))
    return matrix


def baseline_corrected_matrix(
    matrix: np.ndarray,
    baseline: list[float],
    correction: str,
) -> np.ndarray:
    """Apply a layer-wise random-pair baseline correction to the matrix."""
    baseline_array = np.array(baseline, dtype=float)[:, None]
    if correction == "subtract":
        return matrix - baseline_array
    if correction == "divide":
        return np.divide(
            matrix,
            baseline_array,
            out=np.full_like(matrix, np.nan, dtype=float),
            where=np.abs(baseline_array) > 1e-12,
        )
    raise ValueError(f"Unsupported baseline correction: {correction}")


def layer_transition_vectors(
    line1_embeddings_by_layer: np.ndarray,
    line2_embeddings_by_layer: np.ndarray,
    pca_dims: int | None = None,
) -> np.ndarray:
    """Return transition vectors by layer in the selected embedding space."""
    transition_vectors = []
    for layer_idx in range(line1_embeddings_by_layer.shape[0]):
        line1_coords, line2_coords = pca_project_couplet(
            line1_embeddings_by_layer[layer_idx],
            line2_embeddings_by_layer[layer_idx],
            pca_dims,
        )
        transition_vectors.append(
            np.concatenate(
                [
                    np.diff(line1_coords, axis=0),
                    np.diff(line2_coords, axis=0),
                ],
                axis=0,
            )
        )
    return np.array(transition_vectors)


def save_layer_position_matrix(
    matrix: np.ndarray,
    output_dir: str,
    filename: str,
    value_key: str = "mean_transition_alignment",
    baseline: list[float] | None = None,
    correction: str | None = None,
    pca_dims: int | None = None,
    layer_center_normalize: bool = False,
) -> str:
    """Save a 13 x 4 aggregate matrix as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    payload = {
        "layers": list(LAYERS),
        "positions": list(POSITIONS),
        "pca_dims": pca_dims,
        "layer_center_normalize": layer_center_normalize,
        value_key: matrix.tolist(),
    }
    if baseline is not None:
        payload["random_pair_baseline"] = baseline
        payload["baseline_method"] = "position_matched_random_pair"
    if correction is not None:
        payload["baseline_correction"] = correction
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def save_layer_line_plot(
    matrix: np.ndarray,
    output_dir: str,
    filename: str,
    ylabel: str = "Mean transition alignment (cosine)",
    title: str | None = None,
    layer_label: str = "Encoder layer",
) -> str:
    """Save a line plot with one curve per couplet position across layers."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for position_idx, position in enumerate(POSITIONS):
        ax.plot(
            LAYERS,
            matrix[:, position_idx],
            marker="o",
            linewidth=2.2,
            markersize=4.2,
            color=POSITION_COLORS[position],
            label=position_label(position),
        )

    ax.set_xlabel(layer_label)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_xticks(LAYERS)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.legend(title="Couplet role", frameon=False, ncol=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_inner_outer_stats(
    rows: list[dict],
    output_dir: str,
    filename: str,
    num_permutations: int,
    num_bootstrap: int,
    ci_level: float,
    seed: int,
    p_adjust: str,
    pca_dims: int | None,
    layer_center_normalize: bool,
) -> str:
    """Save inner-vs-outer contrast statistics as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    payload = {
        "contrast": (
            f"mean(position {sorted(INNER_POSITIONS)}) - "
            f"mean(position {sorted(OUTER_POSITIONS)})"
        ),
        "unit": "couplet",
        "layers": list(LAYERS),
        "num_permutations": num_permutations,
        "num_bootstrap": num_bootstrap,
        "ci_level": ci_level,
        "seed": seed,
        "p_adjust": p_adjust,
        "pca_dims": pca_dims,
        "layer_center_normalize": layer_center_normalize,
        "rows": rows,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def save_inner_outer_contrast_plot(
    rows: list[dict],
    output_dir: str,
    filename: str,
    layer_label: str = "Encoder layer",
) -> str:
    """Save a layer-wise inner-minus-outer contrast plot with bootstrap CIs."""
    os.makedirs(output_dir, exist_ok=True)
    layers = np.array([row["layer"] for row in rows], dtype=int)
    means = np.array([row["inner_outer_diff"] for row in rows], dtype=float)
    ci_low = np.array([row["ci_low"] for row in rows], dtype=float)
    ci_high = np.array([row["ci_high"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        layers,
        means,
        marker="o",
        linewidth=2.2,
        markersize=4.2,
        color="#0072B2",
        label="Inner - outer",
    )

    has_ci = ~np.isnan(ci_low) & ~np.isnan(ci_high)
    if np.any(has_ci):
        ax.fill_between(
            layers[has_ci],
            ci_low[has_ci],
            ci_high[has_ci],
            color="#0072B2",
            alpha=0.18,
            label="Bootstrap CI",
        )

    ax.axhline(0, color="#222222", linewidth=1, alpha=0.75)
    ax.set_xlabel(layer_label)
    ax.set_ylabel("Mean transition alignment difference")
    ax.set_xticks(LAYERS)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_results_json(results: dict, output_dir: str, filename: str) -> str:
    """Save detailed per-couplet, per-layer transition alignment results."""
    os.makedirs(output_dir, exist_ok=True)
    records = [
        record
        for position in POSITIONS
        for record in results[position]
    ]

    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return output_path


def results_match_punctuation_setting(results: dict, include_punctuation: bool) -> bool:
    """Return True when cached records were produced with the same context mode."""
    for position in POSITIONS:
        for record in results[position]:
            if record.get("include_punctuation") is not include_punctuation:
                return False
    return True


def results_match_projection_setting(results: dict, pca_dims: int | None) -> bool:
    """Return True when cached records use the requested projection setting."""
    for position in POSITIONS:
        for record in results[position]:
            if record.get("pca_dims") != pca_dims:
                return False
    return True


def results_match_normalization_setting(
    results: dict,
    layer_center_normalize: bool,
) -> bool:
    """Return True when cached records use the requested geometry normalization."""
    for position in POSITIONS:
        for record in results[position]:
            if record.get("layer_center_normalize", False) is not layer_center_normalize:
                return False
    return True


def results_match_model_setting(
    results: dict,
    model_name: str,
    tokenizer_name: str,
) -> bool:
    """Return True when cached records use the requested model/tokenizer."""
    for position in POSITIONS:
        for record in results[position]:
            if (
                record.get("model_name") != model_name
                or record.get("tokenizer_name") != tokenizer_name
            ):
                return False
    return True


def load_results_json(output_dir: str, filename: str) -> dict:
    """Load detailed transition alignment results into position buckets."""
    results_path = os.path.join(output_dir, filename)
    with open(results_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    results = defaultdict(list)
    skipped = 0
    for record in records:
        position = record.get("position")
        layer_values = record.get("layer_mean_alignments")
        if (
            position not in POSITIONS
            or not isinstance(layer_values, list)
            or len(layer_values) != len(LAYERS)
        ):
            skipped += 1
            continue
        results[position].append(record)

    if skipped:
        print(f"Skipped {skipped} malformed cached result records")
    return results


def results_have_required_metrics(results: dict) -> bool:
    """Return True when cached records contain all per-layer alignment scores."""
    for position in POSITIONS:
        records = results[position]
        if not records:
            return False
        for record in records:
            if len(record.get("layer_mean_alignments", [])) != len(LAYERS):
                return False
    return True


def save_random_pair_baseline(
    baseline: list[float],
    output_dir: str,
    filename: str,
    reservoir_size: int,
    pairs_per_transition: int,
    transition_slots: tuple[int, ...],
    seed: int,
    include_punctuation: bool,
    pca_dims: int | None,
    layer_center_normalize: bool,
    model_name: str,
    tokenizer_name: str,
) -> str:
    """Save the layer-wise position-matched random-pair baseline."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    payload = {
        "layers": list(LAYERS),
        "random_pair_baseline": baseline,
        "baseline_method": "position_matched_random_pair",
        "transition_slots": list(transition_slots),
        "line_sides": ["line1", "line2"],
        "reservoir_size": reservoir_size,
        "pairs_per_transition": pairs_per_transition,
        "seed": seed,
        "include_punctuation": include_punctuation,
        "pca_dims": pca_dims,
        "layer_center_normalize": layer_center_normalize,
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def random_pair_baseline_cache_matches(
    output_dir: str,
    filename: str,
    reservoir_size: int,
    pairs_per_transition: int,
    transition_slots: tuple[int, ...],
    seed: int,
    include_punctuation: bool,
    pca_dims: int | None,
    layer_center_normalize: bool,
    model_name: str,
    tokenizer_name: str,
) -> bool:
    """Return True if the cached baseline was made with matching settings."""
    baseline_path = os.path.join(output_dir, filename)
    if not os.path.exists(baseline_path):
        return False

    with open(baseline_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    baseline = payload.get("random_pair_baseline")
    return (
        isinstance(baseline, list)
        and len(baseline) == len(LAYERS)
        and payload.get("baseline_method") == "position_matched_random_pair"
        and payload.get("transition_slots") == list(transition_slots)
        and payload.get("line_sides") == ["line1", "line2"]
        and payload.get("reservoir_size") == reservoir_size
        and payload.get("pairs_per_transition") == pairs_per_transition
        and payload.get("seed") == seed
        and payload.get("include_punctuation") is include_punctuation
        and payload.get("pca_dims") == pca_dims
        and payload.get("layer_center_normalize", False) is layer_center_normalize
        and payload.get("model_name") == model_name
        and payload.get("tokenizer_name") == tokenizer_name
    )


def load_random_pair_baseline(output_dir: str, filename: str) -> list[float]:
    """Load a cached layer-wise random-pair baseline."""
    baseline_path = os.path.join(output_dir, filename)
    with open(baseline_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    baseline = payload.get("random_pair_baseline")
    if not isinstance(baseline, list) or len(baseline) != len(LAYERS):
        raise RuntimeError(f"Malformed random-pair baseline cache: {baseline_path}")
    return [float(value) for value in baseline]


def run_meter_analysis(meter: int, args: argparse.Namespace) -> None:
    """Run the full transition-alignment pipeline for one meter."""
    meter_slug = METER_NAMES[meter]
    transition_slots = transition_slots_for_meter(meter)
    include_punctuation = not args.no_punctuation
    tokenizer_name = args.tokenizer or args.model
    layer_label = f"{args.model_label} layer" if args.model_label else "Encoder layer"

    results_filename = meter_filename(RESULTS_FILENAME, meter_slug)
    baseline_filename = meter_filename(BASELINE_FILENAME, meter_slug)
    cached_results_path = os.path.join(args.output_dir, results_filename)
    cached_baseline_path = os.path.join(args.output_dir, baseline_filename)
    baseline_requested = args.baseline_correction != "none"
    random_pair_baseline = None
    results_loaded_from_cache = False

    print(f"\n{'=' * 72}\nMeter: {meter}言律詩 ({meter_slug})\n{'=' * 72}")

    if os.path.exists(cached_results_path) and not args.override:
        print(f"Loading cached results: {cached_results_path}")
        results = load_results_json(args.output_dir, results_filename)
        if results_have_required_metrics(results):
            punctuation_cache_ok = results_match_punctuation_setting(
                results,
                include_punctuation,
            )
            projection_cache_ok = results_match_projection_setting(
                results,
                args.pca_dims,
            )
            normalization_cache_ok = results_match_normalization_setting(
                results,
                args.layer_center_normalize,
            )
            model_cache_ok = results_match_model_setting(
                results,
                args.model,
                tokenizer_name,
            )
            baseline_cache_ok = (
                not baseline_requested
                or random_pair_baseline_cache_matches(
                    args.output_dir,
                    baseline_filename,
                    args.baseline_reservoir_size,
                    args.baseline_pairs_per_transition,
                    transition_slots,
                    args.seed,
                    include_punctuation,
                    args.pca_dims,
                    args.layer_center_normalize,
                    args.model,
                    tokenizer_name,
                )
            )
            if not punctuation_cache_ok:
                print(
                    "Cached results use a different punctuation setting; "
                    "recomputing."
                )
            elif not projection_cache_ok:
                print(
                    "Cached results use a different projection setting; "
                    "recomputing."
                )
            elif not normalization_cache_ok:
                print(
                    "Cached results use a different centering/normalization "
                    "setting; recomputing."
                )
            elif not model_cache_ok:
                print(
                    "Cached results use a different model/tokenizer; "
                    "recomputing."
                )
            elif baseline_requested and not baseline_cache_ok:
                print(
                    "Cached results found, but the random-pair baseline is "
                    "missing or uses different settings; recomputing."
                )
            else:
                results_loaded_from_cache = True
                if baseline_requested:
                    print(f"Loading cached random-pair baseline: {cached_baseline_path}")
                    random_pair_baseline = load_random_pair_baseline(
                        args.output_dir, baseline_filename
                    )
        else:
            print("Cached results are missing layer alignment fields; recomputing.")

    if not results_loaded_from_cache:
        if args.override and os.path.exists(cached_results_path):
            print(f"Override requested; recomputing cached results at {cached_results_path}")

        device = select_device()
        print(f"Using device: {device}")

        data_path = args.data_path_template.format(meter=meter_slug)
        data = balance_data_by_position(
            load_alignment_data(data_path),
            args.max_samples,
            args.seed,
        )
        print(f"Loaded {len(data)} couplets")
        print(f"Batch size: {args.batch_size} couplets")

        print(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"Loading {args.model_label} encoder: {args.model}")
        model = AutoModel.from_pretrained(
            args.model,
            output_hidden_states=True,
            trust_remote_code=args.trust_remote_code,
        )
        model.to(device)
        model.eval()

        results = defaultdict(list)
        layer_centers = None
        if args.layer_center_normalize:
            print(
                "\nEstimating layer-wise corpus centers for contextual "
                "character embeddings..."
            )
            layer_centers, center_skips = estimate_layer_centers(
                data,
                meter,
                args.batch_size,
                tokenizer,
                model,
                device,
                include_punctuation,
            )
            if center_skips:
                print(f"Skipped {center_skips} items while estimating layer centers")

        baseline_state = (
            init_random_pair_baseline(
                reservoir_size=args.baseline_reservoir_size,
                hidden_size=args.pca_dims or int(model.config.hidden_size),
                transition_slots=transition_slots,
                seed=args.seed,
            )
            if baseline_requested
            else None
        )
        skipped = 0

        print(
            "\nCalculating "
            f"{projection_description(args.pca_dims)} transition alignment "
            f"for layers {LAYERS[0]}-{LAYERS[-1]}..."
        )
        for batch in tqdm(iter_batches(data, args.batch_size), desc="Batches"):
            records, batch_skips = valid_records_for_batch(batch, meter)
            skipped += batch_skips

            if not records:
                continue

            records, texts, target_indices_batch, token_skips = validate_token_lengths(
                records,
                tokenizer,
                include_punctuation,
            )
            skipped += token_skips
            if not records:
                continue

            embeddings_batch = get_character_embeddings_batch(
                texts,
                target_indices_batch,
                tokenizer,
                model,
                device,
            )
            for record, embeddings_by_layer in zip(records, embeddings_batch):
                line_len = len(record["line1"])
                if embeddings_by_layer.shape[1] < line_len * 2:
                    skipped += 1
                    continue

                if layer_centers is not None:
                    embeddings_by_layer = layer_center_normalize_embeddings(
                        embeddings_by_layer,
                        layer_centers,
                    )
                line1_embeddings = embeddings_by_layer[:, :line_len, :]
                line2_embeddings = embeddings_by_layer[:, line_len:line_len * 2, :]
                alignment = layer_transition_alignment(
                    line1_embeddings,
                    line2_embeddings,
                    args.pca_dims,
                )
                if baseline_state is not None:
                    transition_vectors = layer_transition_vectors(
                        line1_embeddings,
                        line2_embeddings,
                        args.pca_dims,
                    )
                    update_random_pair_baseline(
                        baseline_state,
                        transition_vectors,
                        args.baseline_pairs_per_transition,
                    )
                position = record["position"]

                results[position].append(
                    {
                        **record,
                        **alignment,
                        "include_punctuation": include_punctuation,
                        "pca_dims": args.pca_dims,
                        "layer_center_normalize": args.layer_center_normalize,
                        "model_name": args.model,
                        "tokenizer_name": tokenizer_name,
                        "model_label": args.model_label,
                    }
                )

        if skipped:
            print(f"Skipped {skipped} malformed or non-character-tokenized items")

        if baseline_state is not None:
            random_pair_baseline = random_pair_baseline_values(baseline_state)

    matrix = build_layer_position_matrix(results)
    projection = projection_description(args.pca_dims)

    print("\n" + "=" * 72)
    print(f"{projection.upper()} TRANSITION ALIGNMENT BY LAYER AND POSITION ({meter_slug})")
    print("=" * 72)
    print("Transition alignment is the mean cosine similarity of corresponding")
    print("character-to-character movement vectors between the two lines.")
    print(f"Alignment space: {projection}.")
    print(f"Layer 0 is the embedding layer; layers 1-{LAYERS[-1]} are transformer layers.")
    if include_punctuation:
        print("Encoder context includes punctuation: line1，line2。")
    else:
        print("Encoder context excludes punctuation: line1+line2")
    if args.layer_center_normalize:
        print("Geometry normalization: layer-wise corpus centering + L2 token normalization.")
    else:
        print("Geometry normalization: raw contextual embeddings.")
    print(f"Permutations: {args.num_permutations}")
    print(f"Bootstrap resamples: {args.num_bootstrap}")
    print(f"Statistic seed: {args.stat_seed}")

    print("\nMean transition alignment matrix (rows=layers, columns=positions 1-4):")
    header = "Layer  " + "  ".join(f"Pos {position}" for position in POSITIONS)
    print(header)
    for layer_idx in LAYERS:
        values = "  ".join(
            f"{matrix[layer_idx, position_idx]:.4f}"
            for position_idx in range(len(POSITIONS))
        )
        print(f"{layer_idx:>5}  {values}")

    if random_pair_baseline is not None:
        print("\nPosition-matched random-pair baseline by layer:")
        for layer_idx, value in zip(LAYERS, random_pair_baseline):
            print(f"  Layer {layer_idx:>2}: {value:.4f}")

    print("\n" + "-" * 72)
    print("POSITION SUMMARIES BY LAYER")
    print("-" * 72)
    for position in POSITIONS:
        print(f"\nPosition {position} ({io_role(position)})")
        for layer_idx in LAYERS:
            alignments = [
                record["layer_mean_alignments"][layer_idx]
                for record in results[position]
            ]
            print(f"  Layer {layer_idx:>2}: {summarize(alignments)}")

    contrast_stats = compute_inner_outer_stats(
        results,
        args.num_permutations,
        args.num_bootstrap,
        args.ci_level,
        args.stat_seed,
        args.p_adjust,
    )
    print_inner_outer_stats(contrast_stats, args.ci_level, args.p_adjust)

    print("\n" + "=" * 72)
    print(f"LOWEST / HIGHEST FINAL-LAYER ALIGNMENT EXAMPLES BY POSITION ({meter_slug})")
    print("=" * 72)
    final_layer = LAYERS[-1]
    for position in POSITIONS:
        bucket = sorted(
            results[position],
            key=lambda record: record["layer_mean_alignments"][final_layer],
        )
        if not bucket:
            continue

        low = bucket[0]
        high = bucket[-1]
        print(f"\nPosition {position}")
        print(
            f"  Lowest : {low['layer_mean_alignments'][final_layer]:.4f}  "
            f"{low['couplet_traditional']}"
        )
        print(
            f"  Highest: {high['layer_mean_alignments'][final_layer]:.4f}  "
            f"{high['couplet_traditional']}"
        )

    print("\n" + "=" * 72)
    print(f"SAVING OUTPUTS ({meter_slug})")
    print("=" * 72)
    matrix_path = save_layer_position_matrix(
        matrix,
        args.output_dir,
        filename=meter_filename(MATRIX_FILENAME, meter_slug),
        pca_dims=args.pca_dims,
        layer_center_normalize=args.layer_center_normalize,
    )
    line_plot_path = save_layer_line_plot(
        matrix,
        args.output_dir,
        filename=meter_filename(LINE_PLOT_FILENAME, meter_slug),
        title=None,
        layer_label=layer_label,
    )
    contrast_stats_path = save_inner_outer_stats(
        contrast_stats,
        args.output_dir,
        meter_filename(CONTRAST_STATS_FILENAME, meter_slug),
        args.num_permutations,
        args.num_bootstrap,
        args.ci_level,
        args.stat_seed,
        args.p_adjust,
        args.pca_dims,
        args.layer_center_normalize,
    )
    contrast_plot_path = save_inner_outer_contrast_plot(
        contrast_stats,
        args.output_dir,
        filename=meter_filename(CONTRAST_PLOT_FILENAME, meter_slug),
        layer_label=layer_label,
    )
    print(f"Saved matrix: {matrix_path}")
    print(f"Saved transition alignment plot: {line_plot_path}")
    print(f"Saved contrast stats: {contrast_stats_path}")
    print(f"Saved contrast plot: {contrast_plot_path}")

    if random_pair_baseline is not None:
        if results_loaded_from_cache:
            print(f"Reused random-pair baseline: {cached_baseline_path}")
        else:
            baseline_path = save_random_pair_baseline(
                random_pair_baseline,
                args.output_dir,
                baseline_filename,
                args.baseline_reservoir_size,
                args.baseline_pairs_per_transition,
                transition_slots,
                args.seed,
                include_punctuation,
                args.pca_dims,
                args.layer_center_normalize,
                args.model,
                tokenizer_name,
            )
            print(f"Saved random-pair baseline: {baseline_path}")

        corrections = (
            ("subtract", SUBTRACTED_MATRIX_FILENAME, SUBTRACTED_LINE_PLOT_FILENAME,
             "Alignment minus random-pair baseline"),
            ("divide", RATIO_MATRIX_FILENAME, RATIO_LINE_PLOT_FILENAME,
             "Alignment / random-pair baseline"),
        )
        for correction, matrix_filename, plot_filename, ylabel in corrections:
            if args.baseline_correction not in {correction, "both"}:
                continue

            corrected_matrix = baseline_corrected_matrix(
                matrix,
                random_pair_baseline,
                correction,
            )
            corrected_matrix_path = save_layer_position_matrix(
                corrected_matrix,
                args.output_dir,
                filename=meter_filename(matrix_filename, meter_slug),
                value_key=f"baseline_{correction}_transition_alignment",
                baseline=random_pair_baseline,
                correction=correction,
                pca_dims=args.pca_dims,
                layer_center_normalize=args.layer_center_normalize,
            )
            corrected_line_plot_path = save_layer_line_plot(
                corrected_matrix,
                args.output_dir,
                filename=meter_filename(plot_filename, meter_slug),
                ylabel=ylabel,
                title=None,
                layer_label=layer_label,
            )
            print(f"Saved {correction} matrix: {corrected_matrix_path}")
            print(f"Saved {correction} plot: {corrected_line_plot_path}")

    if results_loaded_from_cache:
        print(f"Reused detailed results: {cached_results_path}")
    else:
        results_path = save_results_json(results, args.output_dir, results_filename)
        print(f"Saved detailed results: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute full-embedding transition alignment by encoder layer "
            "and couplet position, for pentasyllabic and/or heptasyllabic "
            "regulated-verse couplets"
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"BERT-style encoder path or Hugging Face identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="Tokenizer path or Hugging Face identifier (default: same as --model)",
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default=DEFAULT_MODEL_LABEL,
        help=f"Short model label used in logs and axis labels (default: {DEFAULT_MODEL_LABEL})",
    )
    parser.add_argument(
        "--meters",
        type=int,
        nargs="+",
        choices=METERS,
        default=list(METERS),
        help="Which meters to evaluate (default: both 5 and 7)",
    )
    parser.add_argument(
        "--data-path-template",
        type=str,
        default=DEFAULT_DATA_PATH_TEMPLATE,
        help=(
            "Template for per-meter alignment data JSON paths; {meter} is "
            f"replaced with 'penta' or 'hepta' (default: {DEFAULT_DATA_PATH_TEMPLATE})"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100000,
        help="Total examples to use per meter, split equally across the four couplet positions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for balancing prepared data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of couplets per embedding batch",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=RESULTS_DIR,
        help="Directory for generated plots and detailed results",
    )
    parser.add_argument(
        "--pca-dims",
        "--PCA",
        dest="pca_dims",
        type=int,
        default=None,
        help=(
            "Project each couplet's Hanzi embeddings to this many PCA dimensions "
            "before computing transition alignment. Defaults to full embeddings."
        ),
    )
    parser.add_argument(
        "--layer-center-normalize",
        action="store_true",
        help=(
            "Subtract each layer's corpus mean contextual character embedding and "
            "L2-normalize character embeddings before computing transition cosines "
            "and random-pair baselines."
        ),
    )
    parser.add_argument(
        "--baseline-correction",
        type=str,
        choices=("none", "subtract", "divide", "both"),
        default="none",
        help=(
            "Optionally compute position-matched random-pair baselines per "
            "layer and save baseline-corrected plots"
        ),
    )
    parser.add_argument(
        "--baseline-reservoir-size",
        type=int,
        default=2048,
        help=(
            "Maximum number of transition vectors retained per layer, line "
            "side, and transition slot for random-pair baseline sampling"
        ),
    )
    parser.add_argument(
        "--baseline-pairs-per-transition",
        type=int,
        default=4,
        help=(
            "Random same-slot, opposite-line comparisons sampled for each "
            "transition vector"
        ),
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=10_000,
        help="Number of independent-label permutations for contrast p-values",
    )
    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=1_000,
        help="Number of stratified bootstrap resamples for contrast CIs",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Bootstrap confidence interval level for contrast CIs",
    )
    parser.add_argument(
        "--stat-seed",
        type=int,
        default=0,
        help="Random seed for permutation and bootstrap statistics",
    )
    parser.add_argument(
        "--p-adjust",
        type=str,
        choices=("none", "holm", "bonferroni"),
        default="holm",
        help="Multiple-comparison correction for layer-wise contrast p-values",
    )
    parser.add_argument(
        "--no-punctuation",
        action="store_true",
        help=(
            "Embed each couplet as line1+line2 without the comma/full stop. "
            "By default, the encoder context is line1，line2。 and metrics "
            "are still computed only over Hanzi characters."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model/tokenizer code from Hugging Face repositories",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Recompute transition alignment even if cached results already exist",
    )
    args = parser.parse_args()
    if args.pca_dims is not None and args.pca_dims <= 0:
        raise ValueError("--pca-dims/--PCA must be a positive integer")
    if args.num_permutations < 0:
        raise ValueError("--num-permutations must be non-negative")
    if args.num_bootstrap < 0:
        raise ValueError("--num-bootstrap must be non-negative")
    if not 0.0 < args.ci_level < 1.0:
        raise ValueError("--ci-level must be between 0 and 1")
    load_fonts()

    for meter in args.meters:
        run_meter_analysis(meter, args)


if __name__ == "__main__":
    main()
