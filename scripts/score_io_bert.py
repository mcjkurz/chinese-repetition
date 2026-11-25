"""Score GujiBERT with the entire second line masked at every position."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from opencc import OpenCC
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from bert_io_common import encode_masked_couplet, load_couplets
from common import METER_NAMES, select_device


DEFAULT_MODEL_PATH = "models/gujibert-io-masked-finetuned-penta"
DEFAULT_DATA_PATH = "data/2-3_regulated_verse/couplets_penta_test.json"
DEFAULT_OUTPUT_DIR = "results/bert_masked_line"


class ScoringDataset(Dataset):
    """Create both prompt conditions for each held-out couplet."""

    def __init__(
        self,
        couplets: list[dict],
        tokenizer,
        meter: int,
        convert_to_traditional: bool,
    ):
        self.examples = []
        self.mask_positions = None
        converter = OpenCC("s2t") if convert_to_traditional else None
        for couplet_index, item in enumerate(couplets):
            actual_type = "inner" if item["label"] == "i" else "outer"
            for prompt in ("i", "o"):
                input_ids, target_ids, positions = encode_masked_couplet(
                    tokenizer, item["couplet"], prompt, meter, converter
                )
                self.mask_positions = positions
                self.examples.append(
                    {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                        "target_ids": torch.tensor(target_ids, dtype=torch.long),
                        "couplet_index": couplet_index,
                        "actual_type": actual_type,
                        "prompt": prompt,
                    }
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def paired_permutation_p_value(
    differences: np.ndarray,
    rng: np.random.Generator,
    num_permutations: int,
) -> float:
    if not len(differences):
        return 1.0
    observed = abs(float(np.mean(differences)))
    extreme = 0
    chunk_size = 100
    for start in range(0, num_permutations, chunk_size):
        count = min(chunk_size, num_permutations - start)
        signs = rng.integers(0, 2, size=(count, len(differences)), dtype=np.int8)
        signs = signs * 2 - 1
        means = (signs * differences).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(means) >= observed))
    return (extreme + 1) / (num_permutations + 1)


def holm_adjust(p_values: list[float]) -> list[float]:
    ordered = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    for rank, (index, p_value) in enumerate(ordered):
        running_max = max(running_max, min((len(p_values) - rank) * p_value, 1.0))
        adjusted[index] = running_max
    return adjusted


def summarize(
    records: list[dict], meter: int, model_path: str, permutations: int
) -> dict:
    position_groups = defaultdict(lambda: defaultdict(list))
    line_values = defaultdict(dict)
    for record in records:
        group = (
            record["actual_type"],
            record["prompt"],
            record["position"],
        )
        position_groups[group]["entropy"].append(record["entropy"])
        position_groups[group]["loss"].append(record["loss"])
        line_key = (
            record["couplet_index"],
            record["actual_type"],
            record["prompt"],
        )
        line_values[line_key][record["position"]] = record

    position_summary = []
    for (actual_type, prompt, position), metrics in sorted(position_groups.items()):
        entropies = np.asarray(metrics["entropy"], dtype=float)
        losses = np.asarray(metrics["loss"], dtype=float)
        position_summary.append(
            {
                "actual_type": actual_type,
                "prompt": prompt,
                "position": position,
                "n_couplets": len(losses),
                "mean_entropy_nats": float(entropies.mean()),
                "sd_entropy_nats": float(entropies.std()),
                "mean_observed_loss_nats": float(losses.mean()),
                "sd_observed_loss_nats": float(losses.std()),
            }
        )

    line_summary = {}
    for (couplet_index, actual_type, prompt), positions in line_values.items():
        if len(positions) != meter:
            raise ValueError(
                f"Couplet {couplet_index} has {len(positions)} scored positions, "
                f"expected {meter}."
            )
        line_summary[(couplet_index, actual_type, prompt)] = {
            metric: float(
                np.mean(
                    [
                        positions[position][metric]
                        for position in range(1, meter + 1)
                    ]
                )
            )
            for metric in ("entropy", "loss")
        }

    condition_summary = []
    for actual_type, prompt in (
        ("inner", "i"),
        ("inner", "o"),
        ("outer", "o"),
        ("outer", "i"),
    ):
        values = {
            metric: [
                line_summary[key][metric]
                for key in line_summary
                if key[1:] == (actual_type, prompt)
            ]
            for metric in ("entropy", "loss")
        }
        mean_loss = float(np.mean(values["loss"]))
        condition_summary.append(
            {
                "actual_type": actual_type,
                "prompt": prompt,
                "n_couplets": len(values["loss"]),
                "mean_entropy_nats": float(np.mean(values["entropy"])),
                "sd_entropy_nats": float(np.std(values["entropy"])),
                "mean_observed_loss_nats": mean_loss,
                "sd_observed_loss_nats": float(np.std(values["loss"])),
                "pseudo_perplexity": float(math.exp(mean_loss)),
            }
        )

    tests = []
    rng = np.random.default_rng(42)
    for actual_type, matched_prompt in (("inner", "i"), ("outer", "o")):
        mismatched_prompt = "o" if matched_prompt == "i" else "i"
        indices = sorted(
            key[0]
            for key in line_summary
            if key[1] == actual_type and key[2] == matched_prompt
        )
        entropy_diffs = np.asarray(
            [
                line_summary[(index, actual_type, matched_prompt)]["entropy"]
                - line_summary[(index, actual_type, mismatched_prompt)]["entropy"]
                for index in indices
            ],
            dtype=float,
        )
        loss_diffs = np.asarray(
            [
                line_summary[(index, actual_type, matched_prompt)]["loss"]
                - line_summary[(index, actual_type, mismatched_prompt)]["loss"]
                for index in indices
            ],
            dtype=float,
        )
        tests.append(
            {
                "actual_type": actual_type,
                "matched_prompt": matched_prompt,
                "mismatched_prompt": mismatched_prompt,
                "n_couplets": len(indices),
                "mean_entropy_difference_matched_minus_mismatched_nats": float(
                    entropy_diffs.mean()
                ),
                "entropy_p_value": paired_permutation_p_value(
                    entropy_diffs, rng, permutations
                ),
                "mean_loss_difference_matched_minus_mismatched_nats": float(
                    loss_diffs.mean()
                ),
                "pseudo_perplexity_ratio_matched_over_mismatched": float(
                    math.exp(loss_diffs.mean())
                ),
                "loss_p_value": paired_permutation_p_value(
                    loss_diffs, rng, permutations
                ),
            }
        )

    p_values = [
        test[key]
        for test in tests
        for key in ("entropy_p_value", "loss_p_value")
    ]
    adjusted = holm_adjust(p_values)
    for test, entropy_adjusted, loss_adjusted in zip(
        tests, adjusted[::2], adjusted[1::2]
    ):
        test["entropy_p_adjusted"] = entropy_adjusted
        test["loss_p_adjusted"] = loss_adjusted

    return {
        "experiment": "masked_second_line_character_prediction",
        "model_path": model_path,
        "meter": meter,
        "input_template": (
            "The input contains the i/o prompt, first line, comma, and all "
            "[MASK] tokens for the second line; no second-line characters "
            "are visible when any position is scored."
        ),
        "metrics_note": (
            "Exponentiated mean masked-character loss is pseudo-perplexity, "
            "not joint sequence perplexity."
        ),
        "n_permutations": permutations,
        "condition_summary": condition_summary,
        "per_position_summary": position_summary,
        "matched_prompt_tests": tests,
    }


def save_position_figure(summary: dict, output_path: Path) -> None:
    colors = {"inner": "#6a4c93", "outer": "#2a9d8f"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, metric, title, ylabel in (
        (axes[0], "entropy", "Predictive entropy", "Entropy (nats)"),
        (axes[1], "loss", "Observed-character loss", "Loss (nats)"),
    ):
        positions = np.arange(1, summary["meter"] + 1)
        width = 0.36
        metric_rows = []
        for actual_type, prompt, offset in (
            ("inner", "i", -width / 2),
            ("outer", "o", width / 2),
        ):
            rows = [
                row
                for row in summary["per_position_summary"]
                if row["actual_type"] == actual_type and row["prompt"] == prompt
            ]
            rows.sort(key=lambda row: row["position"])
            mean_key = (
                "mean_entropy_nats" if metric == "entropy" else "mean_observed_loss_nats"
            )
            sd_key = (
                "sd_entropy_nats" if metric == "entropy" else "sd_observed_loss_nats"
            )
            means = [row[mean_key] for row in rows]
            sds = [row[sd_key] for row in rows]
            bars = axis.bar(
                positions + offset,
                means,
                color=colors[actual_type],
                label=f"{actual_type.capitalize()} ({prompt.upper()} prompt)",
                width=width,
                yerr=sds,
                capsize=3,
                error_kw={"elinewidth": 1.4},
            )
            metric_rows.extend(zip(bars, means, sds))
        max_height = max(mean + sd for _, mean, sd in metric_rows)
        axis.set_ylim(0, max_height * 1.2)
        for bar, mean, sd in metric_rows:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                mean + sd + max_height * 0.025,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_title(title)
        axis.set_xlabel("Second-line character position")
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions)
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend(frameon=False)
    figure.suptitle(
        f"GujiBERT: all second-line characters masked "
        f"({METER_NAMES[summary['meter']]})"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score each second-line character with the whole second line "
            "masked, under both inner and outer prompts."
        )
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--no-convert", action="store_true")
    parser.add_argument("--run-name", default="gujibert_io_masked_finetuned")
    args = parser.parse_args()

    couplets = load_couplets(args.data_path)
    meter = couplets[0]["meter"]
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive.")
        couplets = couplets[: args.max_samples]
    device = select_device()
    print(f"Scoring {len(couplets)} {METER_NAMES[meter]} couplets on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_path).to(device)
    model.eval()
    dataset = ScoringDataset(
        couplets,
        tokenizer,
        meter,
        convert_to_traditional=not args.no_convert,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    records = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Scoring masked positions"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, dataset.mask_positions, :].float()
            log_probs = F.log_softmax(logits, dim=-1)
            probabilities = log_probs.exp()
            entropies = -(probabilities * log_probs).sum(dim=-1).cpu().numpy()
            losses = (
                -log_probs.gather(
                    -1, batch["target_ids"].to(device).unsqueeze(-1)
                )
                .squeeze(-1)
                .cpu()
                .numpy()
            )
            for row in range(input_ids.shape[0]):
                for offset in range(meter):
                    records.append(
                        {
                            "couplet_index": int(batch["couplet_index"][row]),
                            "actual_type": batch["actual_type"][row],
                            "prompt": batch["prompt"][row],
                            "position": offset + 1,
                            "entropy": float(entropies[row, offset]),
                            "loss": float(losses[row, offset]),
                        }
                    )

    summary = summarize(records, meter, args.model_path, args.permutations)
    summary["n_scored_couplets"] = len(couplets)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bert_masked_io_{args.run_name}_{METER_NAMES[meter]}_by_position"
    json_path = output_dir / f"{stem}_stats.json"
    png_path = output_dir / f"{stem}.png"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    save_position_figure(summary, png_path)
    print(f"Saved {json_path} and {png_path}")
    for test in summary["matched_prompt_tests"]:
        print(
            f"{test['actual_type']}: matched-vs-mismatched prompt "
            f"loss difference "
            f"{test['mean_loss_difference_matched_minus_mismatched_nats']:.4f} "
            f"nats, adjusted p={test['loss_p_adjusted']:.4g}"
        )


if __name__ == "__main__":
    main()
