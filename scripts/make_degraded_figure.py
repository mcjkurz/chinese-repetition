"""Entropy profiles of the GujiGPT models fine-tuned on degraded or enciphered couplets.

One row per condition with results, one column per meter with results. Each
panel shows mean next-character entropy at every position of the second line:

- degraded conditions (recombined, wordswap, nonsense, shuffled): the model
  tuned on the condition, I prompt, on the held-out couplets of the condition
  and on real inner couplets; the model tuned on real couplets, I prompt, on
  the same degraded couplets and on real inner couplets; the untuned GujiGPT on
  the degraded couplets;
- ciphers (charcipher, wordcipher): the model tuned on the enciphered corpus on
  enciphered inner couplets (I prompt) and outer couplets (O prompt), next to
  the model tuned on real couplets on real inner (I) and outer (O) couplets;
  the model tuned on real couplets and the untuned GujiGPT on the enciphered
  inner couplets.

Results come from results/degraded_training/decoding_{meter}_{condition}.json
and, for references not stored there, results/couplet_controls/decoding_{meter}.json.
Also writes a table of mean entropy and loss for every scored model, prompt and
set (summary.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from score_degraded_models import CIPHERS, TEST_SETS

METERS = {"penta": "Pentasyllabic", "hepta": "Heptasyllabic"}
RED, BLUE, BLACK, GREY, LIGHT = "#d62728", "#2171b5", "#1a1a1a", "#7f7f7f", "#bdbdbd"


def profile(rows: list[dict], key: str = "entropy") -> np.ndarray:
    return np.array([r[key] for r in rows]).mean(axis=0)


def curves_for(condition: str, scores: dict, reference: dict) -> list[tuple]:
    def ref(key_here: str, key_controls: str):
        return scores.get(key_here) or reference.get(key_controls)

    own = TEST_SETS[condition]
    if condition in CIPHERS:
        spec = [(f"Tuned on {condition}, I prompt, enciphered inner", scores.get("I/cipher_inner"), RED, "-"),
                (f"Tuned on {condition}, O prompt, enciphered outer", scores.get("O/cipher_outer"), RED, "--"),
                ("Tuned on real, I prompt, real inner", reference.get("finetuned/I/inner"), BLUE, "-"),
                ("Tuned on real, O prompt, real outer", reference.get("finetuned/O/outer"), BLACK, "--"),
                ("Tuned on real, I prompt, enciphered inner", scores.get("ref_finetuned/I/cipher_inner"), GREY, ":"),
                ("Untuned, enciphered inner", scores.get("ref_base/none/cipher_inner"), LIGHT, "-")]
    else:
        spec = [(f"Tuned on {condition}, I prompt, {condition} test", scores.get(f"I/{own}"), RED, "-"),
                (f"Tuned on {condition}, I prompt, real inner", scores.get("I/inner"), RED, ":"),
                (f"Tuned on real, I prompt, {condition} test",
                 ref(f"ref_finetuned/I/{own}", f"finetuned/I/{own}"), GREY, "--"),
                ("Tuned on real, I prompt, real inner", reference.get("finetuned/I/inner"), BLUE, "-"),
                (f"Untuned, {condition} test", ref(f"ref_base/none/{own}", f"base/none/{own}"), LIGHT, "-")]
    return [s for s in spec if s[1]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results/degraded_training")
    parser.add_argument("--reference-dir", default="results/couplet_controls")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    found = {(m, c): results_dir / f"decoding_{m}_{c}.json" for m in METERS for c in TEST_SETS
             if (results_dir / f"decoding_{m}_{c}.json").exists()}
    meters = [m for m in METERS if any(k[0] == m for k in found)]
    conditions = [c for c in TEST_SETS if any(k[1] == c for k in found)]
    if not found:
        raise SystemExit(f"No results in {results_dir}")

    fig, axes = plt.subplots(len(conditions), len(meters), figsize=(5.8 * len(meters), 3.2 * len(conditions)),
                             sharey=True, squeeze=False)
    table = ["| Meter | Model | Prompt | Set | Mean entropy | Mean loss |", "|:--|:--|:--|:--|--:|--:|"]
    for col, meter in enumerate(meters):
        ref_path = Path(args.reference_dir) / f"decoding_{meter}.json"
        reference = json.loads(ref_path.read_text(encoding="utf-8"))["scores"] if ref_path.exists() else {}
        for key in ("finetuned/I/inner", "finetuned/O/outer"):
            if key in reference:
                table.append(f"| {meter} | tuned on real | {key.split('/')[1]} | {key.split('/')[2]} | "
                             f"{np.mean(profile(reference[key])):.3f} | {np.mean(profile(reference[key], 'loss')):.3f} |")
        for row, condition in enumerate(conditions):
            ax = axes[row][col]
            path = found.get((meter, condition))
            scores = json.loads(path.read_text(encoding="utf-8"))["scores"] if path else {}
            for key, rows in scores.items():
                parts = key.split("/")
                model = f"tuned on {condition}" if len(parts) == 2 else parts[0].replace("ref_", "") + " (reference)"
                table.append(f"| {meter} | {model} | {parts[-2]} | {parts[-1]} | "
                             f"{np.mean(profile(rows)):.3f} | {np.mean(profile(rows, 'loss')):.3f} |")
            for label, rows, color, style in curves_for(condition, scores, reference) if scores else []:
                values = profile(rows)
                ax.plot(np.arange(1, len(values) + 1), values, color=color, linestyle=style, linewidth=2,
                        marker="o", markersize=3.5, label=label)
            ax.set_xticks(range(1, 8 if meter == "hepta" else 6))
            ax.set_title(f"{METERS[meter]}: {condition}", fontsize=10, loc="left")
            ax.set_xlabel("Position in the second line")
            ax.grid(alpha=0.25)
            if col == 0:
                ax.set_ylabel("Next-character entropy (nats)")
            if scores:
                ax.legend(frameon=False, fontsize=6.5, loc="best")
    fig.tight_layout()
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / "entropy_profiles.png", dpi=200, bbox_inches="tight")
    (results_dir / "summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table))
    print(f"Saved: {results_dir / 'entropy_profiles.png'}, {results_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
