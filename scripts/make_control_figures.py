"""Figures and synthesis table for the control conditions of the four experiments.

- results/couplet_controls/entropy_by_position.png: mean next-character entropy at
  every position of the second line for real inner couplets, composed
  couplets, segment-recombined (segswap), character-recombined (pairswap) and
  shuffled (charshuf) couplets under the inner prompt, and real outer
  couplets under the outer prompt (95% confidence intervals over
  couplets, at most ±0.09 nats, are not drawn; score_couplet_controls.py).
- results/couplet_controls/alignment_by_layer.png: transition alignment minus
  the random-pair baseline at every GujiBERT layer for the same six sets
  (align_couplet_controls.py).
- results/control_synthesis.json/.md: for every experiment, the AUC with
  which its measure separates the real texts from each control, so that the
  four experiments can be compared in the same terms. The AUC is the
  probability that a real text shows more of the repetition effect than a
  control text (0.5: indistinguishable).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qhchina import load_fonts
from sklearn.metrics import roc_auc_score

import palette as P

# Colours from the article's palette (palette.py). The two controls that break the real pairs are drawn
# dashed in the per-layer figure.
SETS = {"inner": ("Real inner couplets (I prompt)", P.REAL, "finetuned/I"),
        "composed": ("Composed couplets (I prompt)", P.COMPOSED, "finetuned/I"),
        "segswap": ("Segment-recombined couplets (I prompt)", P.RECOMBINED, "finetuned/I"),
        "pairswap": ("Character-recombined couplets (I prompt)", P.RECOMBINED_LIGHT, "finetuned/I"),
        "charshuf": ("Shuffled couplets (I prompt)", P.SHUFFLED, "finetuned/I"),
        "outer": ("Real outer couplets (O prompt)", P.WITHOUT_PATTERN, "finetuned/O")}
DASHED = {"pairswap", "charshuf"}
METERS = {"penta": "Pentasyllabic", "hepta": "Heptasyllabic"}
OUT = Path("results/couplet_controls")


def auc(higher: np.ndarray, lower: np.ndarray) -> float:
    return float(roc_auc_score([1] * len(higher) + [0] * len(lower), np.r_[higher, lower]))


def ci(values: np.ndarray) -> np.ndarray:
    return 1.96 * values.std(axis=0, ddof=1) / np.sqrt(len(values))


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25)


# Curves of the entropy figure (Fig. 2): inner couplets and the sets built on them under the I prompt
# (solid), with the real outer couplets under the O prompt (dashed) as the reference. Real and composed
# couplets are scored by the model fine-tuned on real couplets (decoding_{meter}.json); nonsense
# couplets by the model fine-tuned on the enciphered corpus (scripts/cipher_finetune/finetune_cipher.py,
# results/cipher_finetune/{meter}/{meter}_cipher_epoch{1,4}.json), after its first and last epoch (dashed).
# Colours from palette.py, as in SETS; the nonsense couplets after the first epoch in a lighter red.
ENTROPY_CURVES = [
    ("Real inner couplets (I)", P.REAL, "-", ("real", "finetuned/I/inner")),
    ("Composed inner couplets (I)", P.COMPOSED, "-", ("real", "finetuned/I/composed")),
    ("Real outer couplets (O)", P.WITHOUT_PATTERN, "-", ("real", "finetuned/O/outer")),
    ("Nonsense inner couplets (I), epoch 1", P.NONSENSE_LIGHT, "--", ("cipher1", "I/inner")),
    ("Nonsense inner couplets (I), epoch 4", P.NONSENSE, "--", ("cipher4", "I/inner")),
]
CIPHER_DIR = Path("results/cipher_finetune")


def entropy_figure(decoding: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True, gridspec_kw={"width_ratios": [5, 7]})
    for ax, meter in zip(axes, METERS):
        sources = {"real": decoding[meter]}
        for epoch in (1, 4):
            path = CIPHER_DIR / meter / f"{meter}_cipher_epoch{epoch}.json"
            sources[f"cipher{epoch}"] = json.loads(path.read_text(encoding="utf-8"))["scores"]
        for label, color, dash, (source, key) in ENTROPY_CURVES:
            values = np.array([r["entropy"] for r in sources[source][key]])
            x = np.arange(1, values.shape[1] + 1)
            # 95% CIs over 1,000 couplets are at most ±0.09 nats, so they are
            # stated in the caption rather than drawn.
            ax.plot(x, values.mean(axis=0), color=color, linewidth=2, marker="o", markersize=4, label=label,
                    linestyle=dash)
        ax.set_xticks(x)
        ax.set_xlabel("Position in the second line")
        ax.set_title(METERS[meter], fontsize=11, loc="left")
        style(ax)
    axes[0].set_ylabel("Next-character entropy (nats)")
    fig.tight_layout()
    # Below the panels, since five curves leave no empty corner.
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=12, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               handlelength=4.5)
    fig.savefig(OUT / "entropy_by_position.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def alignment_figure(alignment: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, meter in zip(axes, METERS):
        for name, (label, color, _) in SETS.items():
            values = np.array([r["alignment"] for r in alignment[meter][name]])
            layers = np.arange(values.shape[1])
            mean = values.mean(axis=0)
            ax.plot(layers, mean, color=color, linewidth=2, label=label.split(" (")[0],
                    linestyle="--" if name in DASHED else "-")
            ax.fill_between(layers, mean - ci(values), mean + ci(values), color=color, alpha=0.18, linewidth=0)
        ax.axhline(0, color="0.6", linewidth=0.6)
        ax.set_xlabel("GujiBERT layer")
        ax.set_title(METERS[meter], fontsize=11, loc="left")
        style(ax)
    axes[0].set_ylabel("Alignment minus random-pair baseline")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "alignment_by_layer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


COMPARISON_TICKS = {"inner": "Real\ninner", "composed": "Composed", "segswap": "Segment-\nrecombined",
                    "pairswap": "Character-\nrecombined", "charshuf": "Shuffled", "outer": "Real\nouter"}


def comparison_figure(alignment: dict, layer: int = 8) -> None:
    """Every couplet of the six sets at one layer, as panel (b) of the essay figure (baguwen_pseudo_legs.py):
    one dot per couplet, bar = mean, whisker = 95% confidence interval."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, meter in zip(axes, METERS):
        for position, (name, (_, color, _)) in enumerate(SETS.items()):
            values = np.array([r["alignment"][layer] for r in alignment[meter][name]])
            jitter = np.random.default_rng(position).uniform(-0.22, 0.22, len(values))
            ax.scatter(position + jitter, values, s=4, color=color, alpha=0.25, linewidths=0, zorder=2)
            # 95% CIs over 1,000 couplets are at most ±0.007, too small to draw.
            ax.hlines(values.mean(), position - 0.3, position + 0.3, color="black", linewidth=1.5, zorder=3)
        ax.axhline(0, color="0.6", linewidth=0.6)
        ax.set_xticks(range(len(SETS)), [COMPARISON_TICKS[n] for n in SETS], fontsize=9)
        ax.set_title(METERS[meter], fontsize=11, loc="left")
        style(ax)
        ax.grid(False, axis="x")
    axes[0].set_ylabel(f"Alignment minus random-pair baseline\n(GujiBERT layer {layer})")
    fig.tight_layout()
    fig.savefig(OUT / f"alignment_layer{layer}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# Panel (a) of the combined figure redraws Fig. 4 (transition_analysis.py) from its saved matrices: the inner
# couplets in two shades of the blue of the real inner couplets of panel (b), the outer couplets in two
# shades of the green of the real outer couplets, dashed.
ROLE_CURVES = {1: ("Head (首聯)", P.WITHOUT_PATTERN_DARK, "o", "--"), 2: ("Jaw (頷聯)", P.REAL_DARK, "o", "-"),
               3: ("Neck (頸聯)", P.REAL, "o", "-"), 4: ("Tail (尾聯)", P.WITHOUT_PATTERN_LIGHT, "o", "--")}
ROLE_MATRIX = "results/transition_gujibert_paper/transition_alignment_layer_position_matrix_baseline_subtracted_{meter}.json"
# Panel (b): the real and artificial couplets of align_couplet_controls.py read by GujiBERT, and the test couplets
# enciphered and read by the cipher model (scripts/cipher_bert/align_eval.py), each minus its own model's baseline.
CIPHER_ALIGNMENT = "results/cipher_bert/alignment/{meter}.json"
CIPHER_MODEL = "Cipher model, enciphered text"
PANEL_B = [("inner", "Real\ninner", P.REAL), ("cipher:inner", "Nonsense\ninner", P.NONSENSE),
           ("outer", "Real\nouter", P.WITHOUT_PATTERN), ("cipher:outer", "Nonsense\nouter", P.NONSENSE_LIGHT),
           ("composed", "Composed\ninner", P.COMPOSED)]


def panel_b_values(alignment: dict, meter: str, key: str, layer: int) -> np.ndarray:
    if key.startswith("cipher:"):
        data = json.loads(Path(CIPHER_ALIGNMENT.format(meter=meter)).read_text(encoding="utf-8"))[CIPHER_MODEL]
        kind = key.split(":")[1]
        return np.array([row[layer] for row, label in zip(data["corrected"], data["labels"]) if label == kind])
    return np.array([r["alignment"][layer] for r in alignment[meter][key]])


def combined_figure(alignment: dict, layer: int = 8) -> None:
    """(a) Alignment by layer for the four couplet roles (Fig. 4); (b) every couplet of the sets of PANEL_B at one layer."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), sharey="row")
    for column, meter in enumerate(METERS):
        ax = axes[0, column]
        data = json.loads(Path(ROLE_MATRIX.format(meter=meter)).read_text(encoding="utf-8"))
        matrix = np.array(data["baseline_subtract_transition_alignment"])
        positions = data["positions"]
        for position, (label, color, marker, dash) in ROLE_CURVES.items():
            ax.plot(data["layers"], matrix[:, positions.index(position)], color=color, linewidth=2, marker=marker,
                    markersize=4, linestyle=dash, label=label)
        ax.axvline(layer, color="0.5", linewidth=0.8, linestyle=":", zorder=0)
        ax.axhline(0, color="0.6", linewidth=0.6)
        ax.set_xticks(data["layers"])
        ax.set_xlabel("GujiBERT layer")
        ax.set_title(("(a) Couplet roles by layer: " + METERS[meter].lower()) if column == 0 else METERS[meter],
                     fontsize=11, loc="left")
        style(ax)

        ax = axes[1, column]
        for index, (key, _, color) in enumerate(PANEL_B):
            values = panel_b_values(alignment, meter, key, layer)
            jitter = np.random.default_rng(index).uniform(-0.22, 0.22, len(values))
            ax.scatter(index + jitter, values, s=4, color=color, alpha=0.25, linewidths=0, zorder=2)
            # 95% CIs over 1,000 couplets are at most ±0.007, too small to draw.
            ax.hlines(values.mean(), index - 0.3, index + 0.3, color="black", linewidth=1.5, zorder=3)
        ax.axhline(0, color="0.6", linewidth=0.6)
        ax.set_xticks(range(len(PANEL_B)), [tick for _, tick, _ in PANEL_B], fontsize=9)
        ax.set_title((f"(b) Every couplet at layer {layer}: " + METERS[meter].lower()) if column == 0
                     else METERS[meter], fontsize=11, loc="left")
        style(ax)
        ax.grid(False, axis="x")
    axes[0, 0].legend(frameon=False, fontsize=9, title="Couplet role", title_fontsize=9, handlelength=4.5)
    for row in axes:
        row[0].set_ylabel("Alignment minus random-pair baseline")
    fig.tight_layout(h_pad=2.5)
    fig.savefig(OUT / "alignment_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def mengzi_declines(path: str) -> np.ndarray:
    """Relative decline of surprisal from the first to the second half, per passage (GujiGPT)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for passage in data.get("passages", [data]):
        s = {p["position"]: p["surprisal_nats"] for p in passage["points"] if p["surprisal_nats"] is not None}
        first = np.mean([s[i] for i in range(2, 33) if i in s])
        second = np.mean([s[i] for i in range(33, 64) if i in s])
        out.append((first - second) / first)
    return np.array(out)


def synthesis(decoding: dict, alignment: dict) -> dict:
    rows = []
    composed, nonsense, scrambled = (mengzi_declines(f"results/surprisal/controls_{k}.json")
                                     for k in ("sensical", "nonsense", "scrambled"))
    rows.append({"experiment": "Prose decoding (Mengzi): relative decline of surprisal, GujiGPT",
                 "real_text": "composed passages (the Mengzi passage itself is one text)",
                 "vs_composed": None, "vs_recombined": None, "vs_nonsense": auc(composed, nonsense),
                 "vs_shuffled": None, "vs_form_broken": auc(composed, scrambled),
                 "form_broken": "scrambled passages", "n": 100})
    for meter in METERS:
        entropy = {n: np.array([np.mean(r["entropy"]) for r in decoding[meter][f"{m}/{n}"]])
                   for n, (_, _, m) in SETS.items()}
        rows.append({"experiment": f"Verse decoding ({METERS[meter].lower()}): next-character entropy, fine-tuned GujiGPT",
                     "real_text": "real inner couplets",
                     "vs_composed": auc(entropy["composed"], entropy["inner"]),
                     "vs_recombined": auc(entropy["segswap"], entropy["inner"]),
                     "vs_nonsense": auc(entropy["pairswap"], entropy["inner"]),
                     "vs_shuffled": auc(entropy["charshuf"], entropy["inner"]),
                     "vs_form_broken": auc(entropy["outer"], entropy["inner"]),
                     "form_broken": "real outer couplets", "n": 1000})
    for meter in METERS:
        aligned = {n: np.array([r["alignment"][8] for r in alignment[meter][n]]) for n in SETS}
        rows.append({"experiment": f"Verse encoding ({METERS[meter].lower()}): transition alignment, GujiBERT layer 8",
                     "real_text": "real inner couplets",
                     "vs_composed": auc(aligned["inner"], aligned["composed"]),
                     "vs_recombined": auc(aligned["inner"], aligned["segswap"]),
                     "vs_nonsense": auc(aligned["inner"], aligned["pairswap"]),
                     "vs_shuffled": auc(aligned["inner"], aligned["charshuf"]),
                     "vs_form_broken": auc(aligned["inner"], aligned["outer"]),
                     "form_broken": "real outer couplets", "n": 1000})
    pairs = json.loads(Path("results/baguwen_regular100/pair_alignment/pairs.json").read_text(encoding="utf-8"))

    def essay_means(label: str) -> np.ndarray:
        by_text: dict[str, list[float]] = {}
        for r in pairs[label]:
            by_text.setdefault(r["text"], []).append(r["alignment"][8] - r["baseline"][8])
        return np.array([np.mean(v) for v in by_text.values()])

    real, comp, rec, non, shuf, prose = (essay_means(k) for k in
                                         ("欽定四書文", "Composed", "Recombined", "Nonsense", "Shuffled", "古文觀止"))
    rows.append({"experiment": "Prose encoding (eight-legged essays): leg alignment, GujiBERT layer 8",
                 "real_text": "regular essays of 欽定四書文", "vs_composed": auc(real, comp),
                 "vs_recombined": auc(real, rec), "vs_nonsense": auc(real, non), "vs_shuffled": auc(real, shuf),
                 "vs_form_broken": auc(real, prose), "form_broken": "古文觀止 pseudo-essays", "n": 100})
    return {"note": __doc__.split("- results/control_synthesis")[1].split("\n\"\"\"")[0].strip(), "rows": rows}


def main() -> None:
    load_fonts()
    decoding = {m: json.loads((OUT / f"decoding_{m}.json").read_text(encoding="utf-8"))["scores"] for m in METERS}
    alignment = {m: json.loads((OUT / f"alignment_{m}.json").read_text(encoding="utf-8"))["alignment"] for m in METERS}
    entropy_figure(decoding)
    alignment_figure(alignment)
    combined_figure(alignment)
    table = synthesis(decoding, alignment)
    Path("results/control_synthesis.json").write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n",
                                                      encoding="utf-8")
    fmt = lambda v: "—" if v is None else f"{v:.2f}"
    columns = ("composed", "recombined", "nonsense", "shuffled", "form_broken")
    lines = ["| Experiment | " + " | ".join(f"vs {c.replace('_', ' ')}" for c in columns) + " |",
             "|:--|" + ":--:|" * len(columns)]
    lines += [f"| {r['experiment']} | " + " | ".join(fmt(r[f'vs_{c}']) for c in columns) + " |" for r in table["rows"]]
    Path("results/control_synthesis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
