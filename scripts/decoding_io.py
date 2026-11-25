"""
Experiment: Decoding behavior for inner vs outer regulated-verse couplets,
for both pentasyllabic and heptasyllabic meters.

For each second-line character position, the script scores the model's
next-token distribution after prompts such as:

    <condition><first line>，

where <condition> is "i" or "o", the prefix of the meter-specific model
trained by finetune_io_gpt.py. Later positions are
evaluated with teacher forcing by adding the true previous second-line
characters to the prefix. The script reports, separately for each meter:

The untuned base model can also be evaluated with --prompt-mode none. In that
mode, the prefix is only the first line and comma, followed by the true
preceding second-line characters. Inner and outer scores are compared within
poem, with paired sign-flip tests and poem-bootstrap confidence intervals.

1. Matched/mismatched prompt summaries by actual couplet type, including
   entropy, observed loss, their standard deviations, and perplexity,
   averaged across the whole second line. Perplexity here is exp(mean
   loss) over many tokens, the standard corpus-level definition.
2. Per-position entropy and observed loss (mean and SD) for aligned cases
   only: true inner + i prompt and true outer + o prompt. Per-position
   results deliberately omit perplexity: with only a handful of couplets
   contributing to each position, a per-position "perplexity" would
   effectively encourage comparing exp(loss) values as if they could be
   averaged or given an SD on the perplexity scale, which is not valid
   since exp() is nonlinear (Jensen's inequality). Loss lives on the
   additive scale the mean/SD were computed on, so it is reported instead.
3. Two PNG figures per meter: aligned inner/outer entropy by second-line
   position, and aligned inner/outer observed loss (with SD) by
   second-line position.

The source data is simplified Chinese, while the default fine-tuned GujiGPT
IO model uses traditional Chinese, so each couplet is converted with OpenCC
before tokenizing.
"""

import argparse
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import opencc
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from common import (
    METER_NAMES,
    ensure_padding_token,
    format_optional_float,
    select_device,
    split_couplet,
)

CONDITIONS = ("i", "o")
POSITIONS = (1, 2, 3, 4)
INNER_POSITIONS = (2, 3)
METERS = (5, 7)
RESULTS_DIR = "results/gpt_decoding"
DEFAULT_FINETUNED_MODEL_PATH = "models/gujigpt-io-finetuned-penta"
DEFAULT_BASE_MODEL_PATH = "hsc748NLP/GujiGPT_fan"
DEFAULT_TOKENIZER_PATH = None
INNER_COLOR = "#6a4c93"
OUTER_COLOR = "#2a9d8f"


def char_positions_for_meter(meter: int) -> tuple[int, ...]:
    """Return the 1-indexed second-line character positions for a meter."""
    return tuple(range(1, meter + 1))


def prompt_prefix(condition: str) -> str:
    """Return the inner/outer prefix token, 'i' or 'o'."""
    return condition


def actual_couplet_type(position: int) -> str:
    """Return the true couplet type from its poem position."""
    return "inner" if position in INNER_POSITIONS else "outer"


def expected_prompt(actual_type: str) -> str:
    """Return the matching prompt condition for an actual couplet type."""
    return "i" if actual_type == "inner" else "o"


def iter_batches(items: list[dict], batch_size: int):
    """Yield consecutive item batches."""
    for start in range(0, len(items), batch_size):
        stop = min(start + batch_size, len(items))
        yield [(index, items[index]) for index in range(start, stop)]


def safe_exp(value: float) -> float:
    """Exponentiate a loss value without overflowing."""
    if np.isnan(value):
        return np.nan
    return math.exp(value) if value < 700 else math.inf


def mean_or_nan(values: list[float]) -> float:
    """Return the mean of values, or NaN for an empty sequence."""
    return float(np.mean(values)) if values else np.nan


def std_or_nan(values: list[float]) -> float:
    """Return the standard deviation of values, or NaN for an empty sequence."""
    return float(np.std(values)) if values else np.nan


def significance_stars(p_value: float) -> str:
    """Return conventional star notation for p-values."""
    if np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original order."""
    adjusted = [np.nan] * len(p_values)
    valid = [(idx, p_value) for idx, p_value in enumerate(p_values) if not np.isnan(p_value)]
    ordered = sorted(valid, key=lambda item: item[1])
    running_max = 0.0
    m = len(ordered)

    for rank, (idx, p_value) in enumerate(ordered):
        corrected = min((m - rank) * p_value, 1.0)
        running_max = max(running_max, corrected)
        adjusted[idx] = running_max

    return adjusted


def paired_permutation_p_value(
    diffs: np.ndarray,
    rng: np.random.Generator,
    num_permutations: int,
    chunk_size: int = 100,
) -> float:
    """Return a two-sided paired sign-flip permutation p-value."""
    if len(diffs) == 0 or num_permutations <= 0:
        return np.nan

    observed = abs(float(np.mean(diffs)))
    extreme_count = 0
    for start in range(0, num_permutations, chunk_size):
        stop = min(start + chunk_size, num_permutations)
        signs = rng.integers(
            0,
            2,
            size=(stop - start, len(diffs)),
            dtype=np.int8,
        )
        signs = (signs * 2) - 1
        permuted_means = (signs * diffs).mean(axis=1)
        extreme_count += int(np.sum(np.abs(permuted_means) >= observed))

    return (extreme_count + 1.0) / (num_permutations + 1.0)


def sanitize_name(name: str) -> str:
    """Return a filename-safe version of a short run label."""
    return "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in name.strip()
    )


def output_stem(run_name: str | None, test_name: str, meter: int) -> str:
    """Return the output filename stem for plots and JSON."""
    test_slug = sanitize_name(test_name.replace("-", "_"))
    meter_slug = METER_NAMES[meter]
    stem = f"decoding_io_{test_slug}_{meter_slug}_aligned_by_position"
    if run_name:
        stem = f"decoding_io_{sanitize_name(run_name)}_{meter_slug}_aligned_by_position"
    return stem


def tokenize_without_specials(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> list[int]:
    """Tokenize text as raw causal-LM ids."""
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def get_next_token_scores(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prefixes: list[str],
    targets: list[str],
    device: torch.device,
) -> list[dict]:
    """
    Return entropy and mean target loss for each prefix/target.

    Entropy is computed over the next-token distribution after the prefix. The
    target character is scored with teacher forcing; if it splits into multiple
    tokens, observed surprisal is averaged across those target tokens.
    """
    encoded_examples = []
    max_length = 0
    for prefix, target in zip(prefixes, targets):
        prefix_ids = tokenize_without_specials(tokenizer, prefix)
        target_ids = tokenize_without_specials(tokenizer, target)
        input_ids = prefix_ids + target_ids
        encoded_examples.append(
            {
                "input_ids": input_ids,
                "prefix_len": len(prefix_ids),
                "target_len": len(target_ids),
            }
        )
        max_length = max(max_length, len(input_ids))

    if not encoded_examples:
        return []

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must have a pad token before batching.")

    input_ids_tensor = torch.full(
        (len(encoded_examples), max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(encoded_examples), max_length),
        dtype=torch.long,
        device=device,
    )

    for row, example in enumerate(encoded_examples):
        ids = example["input_ids"]
        if not ids:
            continue
        length = len(ids)
        input_ids_tensor[row, :length] = torch.tensor(
            ids,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row, :length] = 1

    with torch.no_grad():
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask)

    scores = []
    for row, example in enumerate(encoded_examples):
        prefix_len = example["prefix_len"]
        target_len = example["target_len"]
        if prefix_len == 0 or target_len == 0:
            scores.append({"entropy": np.nan, "loss": np.nan})
            continue

        next_token_logits = outputs.logits[row, prefix_len - 1, :]
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
        entropy = float(-torch.sum(next_token_probs * next_token_log_probs).item())

        token_losses = []
        for target_offset in range(target_len):
            target_pos = prefix_len + target_offset
            target_token_id = int(input_ids_tensor[row, target_pos].item())
            logits = outputs.logits[row, target_pos - 1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            token_losses.append(float(-log_probs[target_token_id].item()))

        scores.append({"entropy": entropy, "loss": float(np.mean(token_losses))})

    return scores


def collect_metric_by_couplet(
    results: dict,
    actual_type: str,
    prompt: str,
    metric: str,
    char_positions: tuple[int, ...],
) -> dict[int, float]:
    """Collect one per-couplet mean metric across all character positions."""
    by_couplet: dict[int, dict[int, float]] = {}
    for position in POSITIONS:
        for record in results[(position, prompt)]:
            if record["actual_type"] != actual_type:
                continue
            by_couplet.setdefault(record["couplet_index"], {})[
                record["char_position"]
            ] = record[metric]

    values = {}
    for couplet_index, position_values in by_couplet.items():
        if all(
            char_position in position_values
            for char_position in char_positions
        ):
            values[couplet_index] = float(np.mean([
                position_values[char_position]
                for char_position in char_positions
            ]))
    return values


def compute_matched_mismatched_tests(
    results: dict,
    char_positions: tuple[int, ...],
    num_permutations: int,
    seed: int,
) -> list[dict]:
    """Test matched vs mismatched prompts within each actual couplet type."""
    rng = np.random.default_rng(seed)
    tests = []
    for actual_type in ("inner", "outer"):
        matched_prompt = expected_prompt(actual_type)
        mismatched_prompt = "o" if matched_prompt == "i" else "i"

        entropy_matched = collect_metric_by_couplet(
            results, actual_type, matched_prompt, "entropy", char_positions
        )
        entropy_mismatched = collect_metric_by_couplet(
            results, actual_type, mismatched_prompt, "entropy", char_positions
        )
        loss_matched = collect_metric_by_couplet(
            results, actual_type, matched_prompt, "loss", char_positions
        )
        loss_mismatched = collect_metric_by_couplet(
            results, actual_type, mismatched_prompt, "loss", char_positions
        )

        entropy_indices = sorted(set(entropy_matched) & set(entropy_mismatched))
        loss_indices = sorted(set(loss_matched) & set(loss_mismatched))
        entropy_diffs = np.array(
            [
                entropy_matched[couplet_index] - entropy_mismatched[couplet_index]
                for couplet_index in entropy_indices
            ],
            dtype=float,
        )
        loss_diffs = np.array(
            [
                loss_matched[couplet_index] - loss_mismatched[couplet_index]
                for couplet_index in loss_indices
            ],
            dtype=float,
        )
        mean_loss_diff = float(np.mean(loss_diffs)) if len(loss_diffs) else np.nan
        entropy_p_value = paired_permutation_p_value(
            entropy_diffs,
            rng,
            num_permutations,
        )
        loss_p_value = paired_permutation_p_value(
            loss_diffs,
            rng,
            num_permutations,
        )

        tests.append(
            {
                "actual_couplet_type": actual_type.capitalize(),
                "matched_prompt": matched_prompt.upper(),
                "mismatched_prompt": mismatched_prompt.upper(),
                "n_couplets_entropy": len(entropy_diffs),
                "mean_entropy_diff_matched_minus_mismatched": (
                    float(np.mean(entropy_diffs)) if len(entropy_diffs) else np.nan
                ),
                "entropy_p_value": entropy_p_value,
                "entropy_p_adjusted": np.nan,
                "n_couplets_loss": len(loss_diffs),
                "mean_loss_diff_matched_minus_mismatched": mean_loss_diff,
                "perplexity_ratio_matched_over_mismatched": safe_exp(mean_loss_diff),
                "loss_p_value": loss_p_value,
                "loss_p_adjusted": np.nan,
            }
        )
    adjusted = holm_adjust(
        [
            p_value
            for test in tests
            for p_value in (test["entropy_p_value"], test["loss_p_value"])
        ]
    )
    for test, entropy_p_adjusted, loss_p_adjusted in zip(
        tests,
        adjusted[0::2],
        adjusted[1::2],
    ):
        test["entropy_p_adjusted"] = entropy_p_adjusted
        test["loss_p_adjusted"] = loss_p_adjusted
    return tests


def collect_matched_mismatched_rows(
    results: dict,
    tests: list[dict],
) -> list[dict]:
    """Collect matched/mismatched summary statistics by true type and prompt."""
    rows = []
    for actual_type, prompt in (
        ("inner", "i"),
        ("inner", "o"),
        ("outer", "o"),
        ("outer", "i"),
    ):
        records = [
            record
            for position in POSITIONS
            for record in results[(position, prompt)]
            if record["actual_type"] == actual_type
        ]
        entropies = [
            record["entropy"]
            for record in records
            if not np.isnan(record["entropy"])
        ]
        losses = [
            record["loss"]
            for record in records
            if not np.isnan(record["loss"])
        ]
        mean_entropy = mean_or_nan(entropies)
        std_entropy = std_or_nan(entropies)
        mean_loss = mean_or_nan(losses)
        std_loss = std_or_nan(losses)
        test = next(
            item
            for item in tests
            if item["actual_couplet_type"] == actual_type.capitalize()
        )
        rows.append(
            {
                "condition": (
                    "Matched" if prompt == expected_prompt(actual_type) else "Mismatched"
                ),
                "actual_couplet_type": actual_type.capitalize(),
                "prompt": prompt.upper(),
                "n_couplets": len({record["couplet_index"] for record in records}),
                "n_tokens": len(records),
                "mean_entropy": mean_entropy,
                "std_entropy": std_entropy,
                "mean_loss": mean_loss,
                "std_loss": std_loss,
                "mean_observed_surprisal": mean_loss,
                "std_observed_surprisal": std_loss,
                "per_token_perplexity": safe_exp(mean_loss),
                "entropy_significance": significance_stars(test["entropy_p_adjusted"]),
                "perplexity_significance": significance_stars(test["loss_p_adjusted"]),
            }
        )
    return rows


def collect_aligned_position_rows(
    results: dict,
    char_positions: tuple[int, ...],
) -> list[dict]:
    """Collect per-position entropy and observed loss for matched true type and prompt.

    Each row aggregates one (actual couplet type, second-line character
    position) combination over held-out couplets: mean/SD entropy (the
    next-token distribution before the character is revealed) and mean/SD
    observed loss (the negative log probability of the actual character).

    Perplexity is deliberately not reported per position. Perplexity is
    exp(loss), a nonlinear transform, so it has no meaningful SD and should
    never be averaged directly (Jensen's inequality: averaging exponentials
    is not the same as exponentiating an average). Loss lives on the
    additive scale the mean/SD were actually computed on, so it is reported
    instead; corpus-level perplexity (exp of mean loss over many tokens) is
    still reported once, in the whole-line matched/mismatched summary.
    """
    rows = []
    for actual_type, prompt in (("inner", "i"), ("outer", "o")):
        for char_position in char_positions:
            records = [
                record
                for position in POSITIONS
                for record in results[(position, prompt)]
                if record["actual_type"] == actual_type
                and record["char_position"] == char_position
            ]
            entropies = [
                record["entropy"]
                for record in records
                if not np.isnan(record["entropy"])
            ]
            losses = [
                record["loss"]
                for record in records
                if not np.isnan(record["loss"])
            ]
            rows.append(
                {
                    "actual_couplet_type": actual_type.capitalize(),
                    "prompt": prompt.upper(),
                    "char_position": char_position,
                    "n_couplets": len({record["couplet_index"] for record in records}),
                    "mean_entropy": mean_or_nan(entropies),
                    "std_entropy": std_or_nan(entropies),
                    "mean_loss": mean_or_nan(losses),
                    "std_loss": std_or_nan(losses),
                }
            )
    return rows


def collect_no_prompt_summary(results: dict) -> list[dict]:
    """Summarize no-prompt token scores by actual couplet type."""
    rows = []
    for actual_type in ("inner", "outer"):
        records = [
            record
            for position in POSITIONS
            for record in results[(position, "none")]
            if record["actual_type"] == actual_type
        ]
        entropies = [record["entropy"] for record in records]
        losses = [record["loss"] for record in records]
        mean_loss = mean_or_nan(losses)
        rows.append(
            {
                "condition": "No prompt",
                "actual_couplet_type": actual_type.capitalize(),
                "prompt": "None",
                "n_couplets": len({record["couplet_index"] for record in records}),
                "n_tokens": len(records),
                "mean_entropy": mean_or_nan(entropies),
                "std_entropy": std_or_nan(entropies),
                "mean_loss": mean_loss,
                "std_loss": std_or_nan(losses),
                "mean_observed_surprisal": mean_loss,
                "std_observed_surprisal": std_or_nan(losses),
                "per_token_perplexity": safe_exp(mean_loss),
                "entropy_significance": "",
                "perplexity_significance": "",
            }
        )
    return rows


def collect_no_prompt_position_rows(
    results: dict,
    char_positions: tuple[int, ...],
) -> list[dict]:
    """Summarize no-prompt scores by actual type and character position."""
    rows = []
    for actual_type in ("inner", "outer"):
        for char_position in char_positions:
            records = [
                record
                for position in POSITIONS
                for record in results[(position, "none")]
                if record["actual_type"] == actual_type
                and record["char_position"] == char_position
            ]
            entropies = [record["entropy"] for record in records]
            losses = [record["loss"] for record in records]
            rows.append(
                {
                    "actual_couplet_type": actual_type.capitalize(),
                    "prompt": "None",
                    "char_position": char_position,
                    "n_couplets": len({record["couplet_index"] for record in records}),
                    "mean_entropy": mean_or_nan(entropies),
                    "std_entropy": std_or_nan(entropies),
                    "mean_loss": mean_or_nan(losses),
                    "std_loss": std_or_nan(losses),
                }
            )
    return rows


def compare_no_prompt_types(
    results: dict,
    num_permutations: int,
    seed: int,
    num_bootstrap: int = 10_000,
) -> dict:
    """Compare inner and outer scores paired by poem, averaging each poem's couplets."""
    by_poem_and_type: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for records in results.values():
        for record in records:
            by_poem_and_type[record["poem_id"]][record["actual_type"]].append(
                record
            )

    poem_metrics = {"inner": {}, "outer": {}}
    for poem_id, types in by_poem_and_type.items():
        for actual_type in ("inner", "outer"):
            couplets: dict[int, dict[int, dict]] = defaultdict(dict)
            for record in types.get(actual_type, []):
                couplets[record["couplet_index"]][record["char_position"]] = record
            if not couplets:
                continue
            line_scores = {
                metric: [
                    float(np.mean([row[metric] for row in positions.values()]))
                    for positions in couplets.values()
                    if positions
                ]
                for metric in ("loss", "entropy")
            }
            poem_metrics[actual_type][poem_id] = {
                metric: float(np.mean(values))
                for metric, values in line_scores.items()
                if values
            }

    shared_poems = sorted(
        set(poem_metrics["inner"]) & set(poem_metrics["outer"])
    )
    rng = np.random.default_rng(seed)
    output = {
        "pairing_unit": "poem",
        "n_poems": len(shared_poems),
        "direction": "inner minus outer; negative means lower inner score",
        "metrics": {},
    }
    for metric in ("loss", "entropy"):
        inner = np.asarray(
            [poem_metrics["inner"][poem_id][metric] for poem_id in shared_poems],
            dtype=float,
        )
        outer = np.asarray(
            [poem_metrics["outer"][poem_id][metric] for poem_id in shared_poems],
            dtype=float,
        )
        differences = inner - outer
        bootstrap_means = np.empty(num_bootstrap, dtype=float)
        for start in range(0, num_bootstrap, 100):
            count = min(100, num_bootstrap - start)
            indices = rng.integers(0, len(differences), size=(count, len(differences)))
            bootstrap_means[start : start + count] = differences[indices].mean(axis=1)
        output["metrics"][metric] = {
            "mean_inner": float(inner.mean()),
            "mean_outer": float(outer.mean()),
            "mean_difference_inner_minus_outer": float(differences.mean()),
            "ci_95_percentile": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
            "paired_sign_flip_p_value": paired_permutation_p_value(
                differences,
                rng,
                num_permutations,
            ),
        }
    output["num_permutations"] = num_permutations
    output["num_bootstrap"] = num_bootstrap
    return output


def print_matched_mismatched_table(rows: list[dict], meter: int) -> None:
    """Print matched/mismatched entropy, loss, and perplexity summaries."""
    print("\n" + "=" * 112)
    print(f"MATCHED AND MISMATCHED PROMPT SUMMARY ({meter}言律诗)")
    print("=" * 112)
    header = (
        f"{'Couplet type':<16}"
        f"{'Prompt':<10}"
        f"{'Mean entropy':>16}"
        f"{'SD entropy':>14}"
        f"{'Mean loss':>14}"
        f"{'SD loss':>12}"
        f"{'Perplexity':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        entropy = format_optional_float(row["mean_entropy"])
        entropy_std = format_optional_float(row["std_entropy"])
        loss = format_optional_float(row["mean_loss"])
        loss_std = format_optional_float(row["std_loss"])
        perplexity = format_optional_float(row["per_token_perplexity"], digits=1)
        if row["condition"] == "Matched":
            entropy = f"{entropy}{row['entropy_significance']}"
            perplexity = f"{perplexity}{row['perplexity_significance']}"
        print(
            f"{row['actual_couplet_type']:<16}"
            f"{row['prompt']:<10}"
            f"{entropy:>16}"
            f"{entropy_std:>14}"
            f"{loss:>14}"
            f"{loss_std:>12}"
            f"{perplexity:>14}"
        )
    has_stars = any(
        row["entropy_significance"] or row["perplexity_significance"]
        for row in rows
    )
    if has_stars:
        print("* p < .05, ** p < .01, *** p < .001 for matched vs mismatched prompts.")
    else:
        print("Permutation tests skipped; no significance markers shown.")


def print_aligned_position_table(rows: list[dict], meter: int) -> None:
    """Print per-position entropy and observed loss for matched true type and prompt.

    Perplexity is intentionally omitted here (see collect_aligned_position_rows):
    it is a corpus-level statistic, not a per-position/per-character one.
    """
    print("\n" + "=" * 114)
    print(f"ALIGNED PER-POSITION ENTROPY AND OBSERVED LOSS ({meter}言律诗)")
    print("=" * 114)
    header = (
        f"{'Actual couplet type':<22}"
        f"{'Prompt':<10}"
        f"{'Position':>10}"
        f"{'n_couplets':>12}"
        f"{'Mean entropy':>16}"
        f"{'SD entropy':>14}"
        f"{'Mean loss':>14}"
        f"{'SD loss':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['actual_couplet_type']:<22}"
            f"{row['prompt']:<10}"
            f"{row['char_position']:>10d}"
            f"{row['n_couplets']:>12d}"
            f"{format_optional_float(row['mean_entropy']):>16}"
            f"{format_optional_float(row['std_entropy']):>14}"
            f"{format_optional_float(row['mean_loss']):>14}"
            f"{format_optional_float(row['std_loss']):>12}"
        )


def json_ready(value):
    """Convert numpy scalars and NaN values into JSON-friendly values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def save_stats_json(
    matched_mismatched_rows: list[dict],
    aligned_position_rows: list[dict],
    matched_mismatched_tests: list[dict],
    metadata: dict,
    output_dir: str,
    output_stem: str,
    condition_summary: list[dict] | None = None,
    type_comparison: dict | None = None,
) -> str:
    """Save the reported result tables as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "metric": "next-token entropy, observed loss, and perplexity",
        "entropy_definition": "entropy of the full next-token distribution",
        "observed_loss_definition": "negative log probability of the true next token",
        "observed_surprisal_definition": "negative log probability of the true next token",
        "standard_deviation_definition": (
            "Population standard deviation over token-level records in each row."
        ),
        "perplexity_definition": "exp(mean observed loss)",
        "significance_test": (
            "Two-sided paired sign-flip permutation tests compare matched vs "
            "mismatched prompts within each actual couplet type. Perplexity "
            "significance is tested on the underlying mean loss scale. "
            "Reported significance markers use Holm-adjusted p-values across "
            "the entropy and loss prompt-effect tests."
        ),
        "metadata": metadata,
        "matched_mismatched_rows": matched_mismatched_rows,
        "matched_mismatched_tests": matched_mismatched_tests,
        "condition_summary": (
            condition_summary
            if condition_summary is not None
            else matched_mismatched_rows
        ),
        "no_prompt_type_comparison": type_comparison,
        "aligned_position_rows": aligned_position_rows,
    }
    output_path = os.path.join(output_dir, f"{output_stem}_stats.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, ensure_ascii=False, indent=2)
    return output_path


def save_aligned_entropy_plot(
    rows: list[dict],
    char_positions: tuple[int, ...],
    output_dir: str,
    output_stem: str,
    prompt_mode: str = "io",
) -> str:
    """Save aligned inner/outer entropy by second-line character position."""
    os.makedirs(output_dir, exist_ok=True)
    x_labels = [str(position) for position in char_positions]
    x = np.arange(len(x_labels))
    width = 0.34
    colors = {"Inner": INNER_COLOR, "Outer": OUTER_COLOR}

    values_by_type = {actual_type: [] for actual_type in ("Inner", "Outer")}
    stds_by_type = {actual_type: [] for actual_type in ("Inner", "Outer")}
    for actual_type in values_by_type:
        for char_position in char_positions:
            row = next(
                (
                    row
                    for row in rows
                    if row["actual_couplet_type"] == actual_type
                    and row["char_position"] == char_position
                ),
                None,
            )
            value = row["mean_entropy"] if row else np.nan
            std = row["std_entropy"] if row else np.nan
            values_by_type[actual_type].append(value)
            stds_by_type[actual_type].append(std)

    finite_values = [
        value
        for values in values_by_type.values()
        for value in values
        if np.isfinite(value)
    ]
    finite_error_tops = [
        value + (std if np.isfinite(std) else 0.0)
        for actual_type, values in values_by_type.items()
        for value, std in zip(values, stds_by_type[actual_type])
        if np.isfinite(value)
    ]
    max_label_height = max(finite_error_tops or finite_values) if finite_values else 1.0

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for offset, actual_type in zip((-width / 2, width / 2), ("Inner", "Outer")):
        values = values_by_type[actual_type]
        stds = stds_by_type[actual_type]
        bars = ax.bar(
            x + offset,
            values,
            width,
            yerr=stds,
            label=(
                ("I/inner" if actual_type == "Inner" else "O/outer")
                if prompt_mode == "io"
                else actual_type
            ),
            color=colors[actual_type],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.8,
            error_kw={
                "ecolor": "#333333",
                "elinewidth": 1.0,
                "capsize": 3,
                "capthick": 1.0,
            },
        )
        for bar, entropy, std in zip(bars, values, stds):
            if not np.isfinite(entropy):
                continue
            error_height = std if np.isfinite(std) else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                entropy + error_height + max_label_height * 0.015,
                f"{entropy:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#222222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Second-line character position")
    ax.set_ylabel("Mean next-token entropy")
    ax.legend(
        title="prompt / couplet" if prompt_mode == "io" else "couplet type",
        frameon=False,
        loc="upper right",
    )
    ax.set_ylim(top=max_label_height * 1.14)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_path = os.path.join(output_dir, f"{output_stem}.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_aligned_loss_plot(
    rows: list[dict],
    char_positions: tuple[int, ...],
    output_dir: str,
    output_stem: str,
    prompt_mode: str = "io",
) -> str:
    """Save aligned inner/outer observed loss (with SD) by second-line character position.

    Loss (negative log probability of the actual character, in nats) is
    plotted rather than perplexity because perplexity is an exponential
    transform of loss: averaging already-exponentiated perplexities (or
    attaching a linear-scale SD to them) is not meaningful, since exp() is
    nonlinear (Jensen's inequality). Loss lives on the additive scale the
    mean and SD were actually computed on, so error bars here are valid;
    per_token_perplexity = exp(mean_loss) is still reported in the stats
    JSON and tables as a point estimate, just without an SD of its own.
    """
    os.makedirs(output_dir, exist_ok=True)
    x_labels = [str(position) for position in char_positions]
    x = np.arange(len(x_labels))
    width = 0.34
    colors = {"Inner": INNER_COLOR, "Outer": OUTER_COLOR}

    values_by_type = {actual_type: [] for actual_type in ("Inner", "Outer")}
    stds_by_type = {actual_type: [] for actual_type in ("Inner", "Outer")}
    for actual_type in values_by_type:
        for char_position in char_positions:
            row = next(
                (
                    row
                    for row in rows
                    if row["actual_couplet_type"] == actual_type
                    and row["char_position"] == char_position
                ),
                None,
            )
            value = row["mean_loss"] if row else np.nan
            std = row["std_loss"] if row else np.nan
            values_by_type[actual_type].append(value)
            stds_by_type[actual_type].append(std)

    finite_values = [
        value
        for values in values_by_type.values()
        for value in values
        if np.isfinite(value)
    ]
    finite_error_tops = [
        value + (std if np.isfinite(std) else 0.0)
        for actual_type, values in values_by_type.items()
        for value, std in zip(values, stds_by_type[actual_type])
        if np.isfinite(value)
    ]
    max_label_height = max(finite_error_tops or finite_values) if finite_values else 1.0

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for offset, actual_type in zip((-width / 2, width / 2), ("Inner", "Outer")):
        values = values_by_type[actual_type]
        stds = stds_by_type[actual_type]
        bars = ax.bar(
            x + offset,
            values,
            width,
            yerr=stds,
            label=(
                ("I/inner" if actual_type == "Inner" else "O/outer")
                if prompt_mode == "io"
                else actual_type
            ),
            color=colors[actual_type],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.8,
            error_kw={
                "ecolor": "#333333",
                "elinewidth": 1.0,
                "capsize": 3,
                "capthick": 1.0,
            },
        )
        for bar, loss, std in zip(bars, values, stds):
            if not np.isfinite(loss):
                continue
            error_height = std if np.isfinite(std) else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                loss + error_height + max_label_height * 0.015,
                f"{loss:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#222222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Second-line character position")
    ax.set_ylabel("Mean observed loss (nats)")
    ax.legend(
        title="prompt / couplet" if prompt_mode == "io" else "couplet type",
        frameon=False,
        loc="upper right",
    )
    ax.set_ylim(top=max_label_height * 1.14)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_path = os.path.join(output_dir, f"{output_stem}.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_meter_evaluation(
    meter: int,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    converter: opencc.OpenCC,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Run the full matched/mismatched + aligned-position evaluation for one meter."""
    char_positions = char_positions_for_meter(meter)
    test_data_path = args.test_data_path_template.format(meter=METER_NAMES[meter])

    print(f"\n{'#' * 72}\n# Meter: {meter}言律诗 ({test_data_path})\n{'#' * 72}")
    print("Loading test data...")
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} couplets")

    results = defaultdict(list)
    skipped = 0
    invalid_scores = 0
    target_char_count = len(char_positions)

    print("\nCalculating next-token scores...")
    for batch in tqdm(iter_batches(test_data, args.batch_size), desc="Batches"):
        prompt_records = []

        for couplet_index, item in batch:
            position = item.get("position")
            if position not in POSITIONS:
                skipped += 1
                continue

            couplet = converter.convert(item["couplet"])
            first_line, second_line = split_couplet(couplet)
            if first_line is None or not second_line:
                skipped += 1
                continue
            if len(second_line) < target_char_count:
                skipped += 1
                continue

            item_actual_type = actual_couplet_type(position)
            conditions = CONDITIONS if args.prompt_mode == "io" else ("none",)
            for condition in conditions:
                prefix_token = prompt_prefix(condition) if condition != "none" else ""
                for char_idx in range(target_char_count):
                    prompt_records.append(
                        {
                            "prefix": (
                                prefix_token
                                + first_line
                                + "，"
                                + second_line[:char_idx]
                            ),
                            "target": second_line[char_idx],
                            "couplet_index": couplet_index,
                            "position": position,
                            "actual_type": item_actual_type,
                            "condition": condition,
                            "prompt_token": prefix_token,
                            "poem_id": item["poem_id"],
                            "char_position": char_idx + 1,
                        }
                    )

        if not prompt_records:
            continue

        scores = get_next_token_scores(
            model,
            tokenizer,
            [record["prefix"] for record in prompt_records],
            [record["target"] for record in prompt_records],
            device,
        )

        for record, score in zip(prompt_records, scores):
            if np.isnan(score["loss"]) or np.isnan(score["entropy"]):
                invalid_scores += 1
                continue

            position = record["position"]
            condition = record["condition"]
            results[(position, condition)].append(
                {
                    "position": position,
                    "condition": condition,
                    "prompt_token": record["prompt_token"],
                    "couplet_index": record["couplet_index"],
                    "actual_type": record["actual_type"],
                    "poem_id": record["poem_id"],
                    "char_position": record["char_position"],
                    "target": record["target"],
                    "entropy": score["entropy"],
                    "loss": score["loss"],
                }
            )

    if skipped:
        print(f"Skipped {skipped} malformed items")
    if invalid_scores:
        print(f"Skipped {invalid_scores} targets that could not be scored")

    if args.prompt_mode == "io":
        matched_mismatched_tests = compute_matched_mismatched_tests(
            results,
            char_positions,
            args.num_permutations,
            args.stat_seed,
        )
        matched_mismatched_rows = collect_matched_mismatched_rows(
            results,
            matched_mismatched_tests,
        )
        condition_summary = matched_mismatched_rows
        aligned_position_rows = collect_aligned_position_rows(
            results, char_positions
        )
        type_comparison = None
        print_matched_mismatched_table(matched_mismatched_rows, meter)
    else:
        matched_mismatched_tests = []
        matched_mismatched_rows = []
        condition_summary = collect_no_prompt_summary(results)
        aligned_position_rows = collect_no_prompt_position_rows(
            results, char_positions
        )
        type_comparison = compare_no_prompt_types(
            results,
            args.num_permutations,
            args.stat_seed,
        )
        print("\nNO-PROMPT SUMMARY BY ACTUAL COUPLET TYPE")
        for row in condition_summary:
            print(
                f"{row['actual_couplet_type']}: "
                f"mean loss={row['mean_loss']:.4f} nats; "
                f"perplexity={row['per_token_perplexity']:.2f}; "
                f"mean entropy={row['mean_entropy']:.4f} nats"
            )
        print("\nPAIRED INNER-OUTER COMPARISON BY POEM")
        for metric, values in type_comparison["metrics"].items():
            print(
                f"{metric}: inner - outer="
                f"{values['mean_difference_inner_minus_outer']:.4f}; "
                f"95% CI={values['ci_95_percentile'][0]:.4f} to "
                f"{values['ci_95_percentile'][1]:.4f}; "
                f"p={values['paired_sign_flip_p_value']:.6g}; "
                f"n_poems={type_comparison['n_poems']}"
            )
    print_aligned_position_table(aligned_position_rows, meter)

    print("\n" + "=" * 72)
    print(f"SAVING OUTPUTS ({meter}言律诗)")
    print("=" * 72)
    stem = output_stem(args.run_name, args.test, meter)
    entropy_plot_path = save_aligned_entropy_plot(
        aligned_position_rows,
        char_positions,
        args.output_dir,
        f"{stem}_entropy",
        args.prompt_mode,
    )
    print(f"Saved: {entropy_plot_path}")
    loss_plot_path = save_aligned_loss_plot(
        aligned_position_rows,
        char_positions,
        args.output_dir,
        f"{stem}_loss",
        args.prompt_mode,
    )
    print(f"Saved: {loss_plot_path}")
    stats_path = save_stats_json(
        matched_mismatched_rows,
        aligned_position_rows,
        matched_mismatched_tests,
        {
            "test": args.test,
            "prompt_mode": args.prompt_mode,
            "meter": meter,
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path or args.model_path,
            "num_permutations": args.num_permutations,
            "stat_seed": args.stat_seed,
            "split_unit": (
                "poem"
                if test_data and isinstance(test_data[0], dict) and test_data[0].get("poem_id")
                else "unspecified"
            ),
            "test_couplets": len(test_data),
            "test_poems": len({item.get("poem_id") for item in test_data if item.get("poem_id")}),
        },
        args.output_dir,
        stem,
        condition_summary=condition_summary,
        type_comparison=type_comparison,
    )
    print(f"Saved: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IO decoding entropy and observed surprisal"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=("finetuned", "base-model"),
        default="finetuned",
        help="Evaluation target: fine-tuned IO model or original base GujiGPT",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("io", "none"),
        default="io",
        help=(
            "Use I/O condition tokens (io) or no condition token (none). "
            "The no-prompt mode compares inner and outer scores paired by poem."
        ),
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
        "--model-path",
        type=str,
        default=None,
        help=(
            "Model path or Hugging Face model name. Defaults to "
            f"{DEFAULT_FINETUNED_MODEL_PATH} for --test finetuned and "
            f"{DEFAULT_BASE_MODEL_PATH} for --test base-model."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=DEFAULT_TOKENIZER_PATH,
        help="Tokenizer path or Hugging Face tokenizer name (default: same as --model-path)",
    )
    parser.add_argument(
        "--test-data-path-template",
        type=str,
        default="data/2-3_regulated_verse/couplets_{meter}_test.json",
        help=(
            "Template for per-meter test data JSON paths; {meter} is replaced "
            "with 'penta' or 'hepta' (default: data/2-3_regulated_verse/couplets_{meter}_test.json)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of couplets per batch",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=RESULTS_DIR,
        help="Directory for generated outputs",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional label added to output filenames",
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=10_000,
        help="Number of paired sign-flip permutations for p-values",
    )
    parser.add_argument(
        "--stat-seed",
        type=int,
        default=0,
        help="Random seed for permutation tests",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model/tokenizer code from Hugging Face repositories",
    )
    args = parser.parse_args()

    if args.num_permutations < 0:
        raise ValueError("--num-permutations must be non-negative")
    if args.test == "base-model" and args.prompt_mode == "io":
        args.num_permutations = 0

    if args.model_path is None:
        args.model_path = (
            DEFAULT_BASE_MODEL_PATH
            if args.test == "base-model"
            else DEFAULT_FINETUNED_MODEL_PATH
        )

    device = select_device()
    print(f"Using device: {device}")

    tokenizer_path = args.tokenizer_path or args.model_path
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )
    ensure_padding_token(tokenizer)

    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    converter = opencc.OpenCC("s2t")
    print(f"Batch size: {args.batch_size} couplets")
    print(f"Test mode: {args.test}")
    print(f"Prompt mode: {args.prompt_mode}")
    print(f"Meters: {args.meters}")
    print(f"Permutations: {args.num_permutations}")
    print("Prediction mode: all second-line characters with teacher forcing")

    for meter in args.meters:
        run_meter_evaluation(meter, model, tokenizer, converter, device, args)


if __name__ == "__main__":
    main()
