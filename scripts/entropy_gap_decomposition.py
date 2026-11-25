#!/usr/bin/env python
"""Explainer figure: split the inner-outer entropy gap into intrinsic vs prompt parts.

Reads the aligned-by-position decoding stats from the 2026-10-27 run_experiments.sh
run and writes a two-panel figure to results/gpt_decoding/.

Logic: the fine-tuned model's aligned gap (outer minus inner entropy, position by
position) mixes the couplet text and the I/O prompt. The shuffled-label model's
tokens are meaningless, so its gap is carried by the text alone. The difference
(fine-tuned minus shuffled) is what the prompt adds.

Run from the repo root:  venv/bin/python scripts/entropy_gap_decomposition.py
"""
import json
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "gpt_decoding"


def aligned_gap(meter, model):
    path = RESULTS / f"decoding_io_gujigpt_{model}_{meter}_aligned_by_position_stats.json"
    inner, outer = {}, {}
    for row in json.load(open(path))["aligned_position_rows"]:
        (inner if row["actual_couplet_type"] == "Inner" else outer)[row["char_position"]] = row["mean_entropy"]
    positions = sorted(inner)
    return [outer[p] - inner[p] for p in positions]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, meter, n_pos in zip(axes, ["penta", "hepta"], [5, 7]):
        ft = aligned_gap(meter, "io_finetuned")
        sh = aligned_gap(meter, "io_shuffled")
        prompt = [f - s for f, s in zip(ft, sh)]
        x = range(1, n_pos + 1)
        w = 0.38
        ax.bar([i - w / 2 for i in x], sh, w, label="Intrinsic (text alone, shuffled-label model)")
        ax.bar([i + w / 2 for i in x], prompt, w, label="Added by the I/O prompt")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xlabel("Second-line character position")
        ax.set_ylabel("Outer minus inner entropy (nats)")
        ax.set_title(f"{meter}: gap {round(ft[0], 2)} nats at position 1 = "
                     f"{round(sh[0], 2)} intrinsic + {round(prompt[0], 2)} prompt")
        ax.legend(fontsize=8)
    fig.suptitle("Decomposition of the inner-outer entropy gap (GujiGPT, held-out poems, 2026-10-27 run)", y=1.02)
    fig.tight_layout()
    out = RESULTS / "entropy_gap_decomposition.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
