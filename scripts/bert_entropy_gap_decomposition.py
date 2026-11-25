#!/usr/bin/env python
"""Decompose aligned GujiBERT entropy gaps using shuffled-label controls."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "bert_masked_line"


def aligned_gap(meter: str, run_name: str) -> list[float]:
    path = (
        RESULTS
        / f"bert_masked_io_{run_name}_{meter}_by_position_stats.json"
    )
    with path.open(encoding="utf-8") as file:
        summary = json.load(file)
    if summary["experiment"] != "masked_second_line_character_prediction":
        raise ValueError(f"Unexpected experiment in {path}")

    aligned = {
        (row["actual_type"], row["prompt"], row["position"]): row[
            "mean_entropy_nats"
        ]
        for row in summary["per_position_summary"]
    }
    return [
        aligned[("outer", "o", position)]
        - aligned[("inner", "i", position)]
        for position in range(1, summary["meter"] + 1)
    ]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    meters = ("penta", "hepta")
    output = {
        "experiment": "masked_second_line_character_prediction",
        "measure": "aligned outer/O minus inner/I entropy, in nats",
        "components": {
            "shuffled_label_gap": (
                "Aligned gap from a model fine-tuned with randomized I/O labels."
            ),
            "prompt_residual": (
                "Fine-tuned aligned gap minus shuffled-label aligned gap; "
                "the estimated contribution associated with informative labels."
            ),
        },
        "meters": {},
    }

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, meter in zip(axes, meters):
        fine_tuned = aligned_gap(meter, "gujibert_io_masked_finetuned")
        shuffled = aligned_gap(meter, "gujibert_io_shuffled")
        residual = [full - control for full, control in zip(fine_tuned, shuffled)]
        positions = list(range(1, len(fine_tuned) + 1))
        output["meters"][meter] = {
            "fine_tuned_aligned_gap_nats": fine_tuned,
            "shuffled_label_gap_nats": shuffled,
            "prompt_residual_nats": residual,
        }

        width = 0.38
        axis.bar(
            [position - width / 2 for position in positions],
            shuffled,
            width,
            label="Shuffled-label gap",
        )
        axis.bar(
            [position + width / 2 for position in positions],
            residual,
            width,
            label="Prompt residual",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(positions)
        axis.set_xlabel("Second-line character position")
        axis.set_ylabel("Outer minus inner entropy (nats)")
        axis.set_title(f"{meter}: all second-line characters masked")
        axis.legend(fontsize=8)

    output["caveat"] = (
        "The shuffled-label component is a control-based estimate, not a "
        "causal decomposition. The prompt residual can include interactions "
        "between fine-tuning and the prompt."
    )
    json_path = RESULTS / "bert_masked_entropy_gap_decomposition.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    figure.suptitle("GujiBERT advance-expectation entropy-gap decomposition")
    figure.tight_layout()
    figure_path = RESULTS / "bert_masked_entropy_gap_decomposition.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {json_path} and {figure_path}")


if __name__ == "__main__":
    main()
