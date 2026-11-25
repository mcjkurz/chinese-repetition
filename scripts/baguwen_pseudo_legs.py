"""Parallel legs against non-parallel pseudo-legs: the essay counterpart of inner vs outer couplets.

Real leg pairs come from eight-legged essays, one condition per --corpus
file: by default the 100 essays of 欽定四書文 marked by hand
(prepare_qinding_legs.py), among them Chen Zilong's essay (--highlight), whose
pairs are reported and drawn separately but not counted twice. Control pairs
come from essays built on the same 100 essays, one condition per --controls
file: composed pairs written by hand after each real pair
(make_qinding_composed.py) and, from make_qinding_nonsense.py, recombined
pairs whose corresponding fragments come from other essays, nonsense pairs in
which every character of an essay is replaced by a random one, and shuffled
pairs whose leg B has its fragments permuted. Pairs with at
least --min-fragments fragments per member are used. For every pair the share of characters that
corresponding fragments have in common is recorded, and alignment is also
reported by bins of that share, so that real and pseudo pairs can be compared
at the same level of verbatim overlap. Pseudo-leg pairs come from
non-parallel classical prose (prepare_pseudo_leg_corpora.py): 2n consecutive
fragments of one piece, the first n as pseudo-leg A and the next n as
pseudo-leg B, for the same n as the real pairs; a --prose file of pseudo-essays
matched to the corpus (make_pseudo_essays.py) gives exactly one pseudo-pair
per real pair instead of a sample per n. Windows whose fragments fall
outside the range of the real fragments' lengths (or --min-chars..--max-chars)
are not used, so that pseudo-legs are built from clause-sized fragments too, nor are
windows in which a pseudo-leg repeats a fragment immediately (a zero
transition has no direction).

Every fragment is encoded on its own by GujiBERT and represented by the mean
of its characters' embeddings, as in baguwen_fragment_space.py. The alignment
of a pair is the mean cosine between corresponding fragment-to-fragment
transitions of its two legs, at every layer. The baseline is position-matched
random pairs as in transition_analysis.py: leg A of a pair against leg B of
--random-pairs unrelated pseudo-leg pairs with the same n, sampled from
--baseline-pool (古文觀止), the same pool for every condition.

Pairs from one essay (or one prose piece) are not independent, so every
comparison is also made with the text as the unit: per-text means compared by
Mann-Whitney test and Cohen's d, a bootstrap over texts for the confidence
interval of each mean and of its difference from the reference, and, for
the controls, a Wilcoxon signed-rank test against the real essay each control
essay was built on.

Writes summary statistics (per layer: mean alignment, baseline, difference,
Mann-Whitney test and Cohen's d against the reference prose; at the figure
layer, the text-level comparisons), per-pair values, and a figure. Panel (c)
of the figure shows the --highlight essay at the figure layer: its fragments,
each encoded on its own as above, projected by one PCA of all its fragments
and drawn twice, legs 1, 3, 5, 7 beside legs 2, 4, 6, 8 on the same axes
(the first two components, or three in two cubes with --essay-dims 3), one
colour per pair of legs and, within a leg, light to dark from the first
fragment to the last. With --replot the figure is redrawn
from the summary.json and pairs.json of --output-dir without scoring again.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from qhchina import load_fonts
from scipy.stats import mannwhitneyu, wilcoxon
from transformers import AutoModel, AutoTokenizer

from baguwen_fragment_space import FRAGMENT_SPLIT
from common import select_device
import palette as P
from self_similarity_matrix import DEFAULT_MODEL

QUOTES = re.compile(r"[「」『』《》〈〉“”‘’（）()·\s]")
# The article's palette (palette.py): real essays blue, composed light blue, nonsense (random characters)
# red, recombined (clause pairs from other essays) purple, shuffled (leg B permuted) dark grey,
# non-parallel prose bluish green.
COLORS = {"欽定四書文": P.REAL, "Composed": P.COMPOSED, "Nonsense": P.NONSENSE, "欽定四書文 (100 essays)": P.REAL,
          "Composed (100 essays)": P.COMPOSED, "Nonsense (100 essays)": P.NONSENSE,
          "古文觀止": P.WITHOUT_PATTERN, "明史": "#a4a4ab", "欽定四書文 (152 regular essays)": P.REAL,
          "Composed (152 essays)": P.COMPOSED, "Nonsense (152 essays)": P.NONSENSE,
          "Recombined": P.RECOMBINED, "Shuffled": P.SHUFFLED}
HIGHLIGHT_COLOR = P.REAL_DARK  # navy: one of the real essays, in a darker shade of their blue
# Panel (c): one colour per pair of legs, the same for both partners, so that the two cubes can be compared.
PAIR_COLORS = ["#1f5fa8", "#e07b10", "#2a8a3e", "#c2185b"]
PAIR_NAMES = ["initial (起股)", "middle (中股)", "latter (後股)", "closing (束股)"]
MUTED = "#8a8a8a"


def load_corpus(path: str, min_fragments: int) -> list[tuple[list[str], list[str], str, int]]:
    """Marked leg pairs of a corpus in the format of data/4_eight_legged_essays/qinding_legs.json, with the id of their essay and their
    position in it (1 for the first pair)."""
    essays = json.loads(Path(path).read_text(encoding="utf-8"))["essays"]
    return [([f for f in FRAGMENT_SPLIT.split(pair["a"]) if f], [f for f in FRAGMENT_SPLIT.split(pair["b"]) if f],
             essay["id"], position)
            for essay in essays for position, pair in enumerate(essay["pairs"], 1) if pair["fragments"] >= min_fragments]


def prose_windows(pieces: list[dict], n: int, min_chars: int, max_chars: int) -> list[tuple[list[str], list[str], str, None]]:
    """Non-overlapping runs of 2n consecutive clause-sized fragments, split into pseudo-legs A and B, with the
    title of their piece."""
    windows = []
    for piece in pieces:
        fragments = [f for f in FRAGMENT_SPLIT.split(QUOTES.sub("", piece["text"])) if f]
        start = 0
        while start + 2 * n <= len(fragments):
            window = fragments[start:start + 2 * n]
            # Two identical consecutive fragments would give a zero transition, which has no direction.
            if all(min_chars <= len(f) <= max_chars for f in window) and \
                    all(x != y for x, y in zip(window[:n - 1], window[1:n])) and \
                    all(x != y for x, y in zip(window[n:-1], window[n + 1:])):
                windows.append((window[:n], window[n:], piece["title"], None))
                start += 2 * n
            else:
                start += 1
    return windows


def encode(fragments: list[str], batch_size: int, model, tokenizer, device) -> np.ndarray | None:
    """Mean character embedding of each fragment at every layer, (fragments, layers, dim); None if any
    fragment does not tokenize one token per character."""
    vectors = []
    for start in range(0, len(fragments), batch_size):
        batch = fragments[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        if [length - 2 for length in encoded["attention_mask"].sum(dim=1).tolist()] != [len(f) for f in batch]:
            return None
        with torch.no_grad():
            hidden = torch.stack(model(**encoded, output_hidden_states=True).hidden_states).float().cpu().numpy()
        vectors.extend(hidden[:, i, 1:1 + len(f)].mean(axis=1) for i, f in enumerate(batch))
    return np.stack(vectors)


def unit_steps(vectors: np.ndarray) -> np.ndarray:
    """Unit transitions between consecutive fragments: (fragments, layers, dim) -> (layers, steps, dim)."""
    steps = np.diff(vectors, axis=0).transpose(1, 0, 2)
    return (steps / np.linalg.norm(steps, axis=-1, keepdims=True)).astype(np.float32)


def pair_steps(pairs: list[tuple[list[str], list[str], str, int | None]], args, model, tokenizer, device) -> list[dict]:
    """Encode each pair's fragments and keep the unit transitions of both legs, with the text the pair comes from
    and its position there."""
    records = []
    for index in range(0, len(pairs), args.pairs_per_batch):
        chunk = pairs[index:index + args.pairs_per_batch]
        flat = [f for a, b, _, _ in chunk for f in a + b]
        vectors = encode(flat, args.batch_size, model, tokenizer, device)
        if vectors is None:  # fall back to one pair at a time to drop only the pairs that do not tokenize
            for a, b, text, position in chunk:
                single = encode(a + b, args.batch_size, model, tokenizer, device)
                if single is not None:
                    records.append({"a": a, "b": b, "text": text, "position": position,
                                    "A": unit_steps(single[:len(a)]), "B": unit_steps(single[len(a):])})
            continue
        offset = 0
        for a, b, text, position in chunk:
            records.append({"a": a, "b": b, "text": text, "position": position, "A": unit_steps(vectors[offset:offset + len(a)]),
                            "B": unit_steps(vectors[offset + len(a):offset + len(a) + len(b)])})
            offset += len(a) + len(b)
    return records


def alignment(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Mean cosine between corresponding transitions, per layer."""
    return np.sum(first * second, axis=-1).mean(axis=-1)


def score(records: list[dict], reference: dict[int, list[dict]], count: int, rng: random.Random) -> list[dict]:
    """Alignment and random-pair baseline of each pair, per layer."""
    for record in records:
        pool = [other for other in reference[len(record["a"])] if other is not record]
        others = rng.sample(pool, min(count, len(pool)))
        record["alignment"] = alignment(record["A"], record["B"])
        record["baseline"] = np.mean([alignment(record["A"], other["B"]) for other in others], axis=0)
        record["length_gap"] = float(np.mean([abs(len(x) - len(y)) for x, y in zip(record["a"], record["b"])]))
        record["shared"] = float(np.mean([len(set(x) & set(y)) / max(len(set(x)), len(set(y)))
                                          for x, y in zip(record["a"], record["b"])]))
    return records


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    pooled = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    return float((x.mean() - y.mean()) / pooled)


def text_means(records: list[dict], layer: int) -> dict[str, float]:
    """Mean difference from the baseline of each text's pairs at one layer."""
    by_text: dict[str, list[float]] = {}
    for r in records:
        by_text.setdefault(r["text"], []).append(float(r["alignment"][layer] - r["baseline"][layer]))
    return {text: float(np.mean(values)) for text, values in by_text.items()}


def bootstrap_texts(records: list[dict], layer: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Pair-level mean at one layer for --bootstrap resamples of whole texts."""
    by_text: dict[str, list[float]] = {}
    for r in records:
        by_text.setdefault(r["text"], []).append(float(r["alignment"][layer] - r["baseline"][layer]))
    sums = np.array([sum(v) for v in by_text.values()])
    sizes = np.array([len(v) for v in by_text.values()])
    picks = rng.integers(0, len(sums), size=(count, len(sums)))
    return sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)


def shade(color: str, t: float) -> tuple[float, float, float]:
    """A leg's colour at relative position t among its fragments: light (0) to dark (1)."""
    hue, _, saturation = colorsys.rgb_to_hls(*to_rgb(color))
    return colorsys.hls_to_rgb(hue, 0.66 - 0.36 * t, saturation)


def essay_projection(path: str, essay_id: str, layer: int, args, model, tokenizer, device) -> dict:
    """The fragments of one essay's legs in text order (1A, 1B, 2A, ...), encoded as in pair_steps, and their
    coordinates in the essay's own three principal components at one layer, with each component's share of
    variance."""
    essay = next(e for e in json.loads(Path(path).read_text(encoding="utf-8"))["essays"] if e["id"] == essay_id)
    legs = [[f for f in FRAGMENT_SPLIT.split(pair[key]) if f] for pair in essay["pairs"] for key in ("a", "b")]
    vectors = encode([f for leg in legs for f in leg], args.batch_size, model, tokenizer, device)
    if vectors is None:
        raise ValueError(f"Essay {essay_id} does not tokenize one token per character")
    centered = vectors[:, layer] - vectors[:, layer].mean(axis=0)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    # Each component points towards its largest loading, so that the view does not depend on the sign of the SVD.
    axes = vh[:3] * np.sign(vh[:3][np.arange(3), np.abs(vh[:3]).argmax(axis=1)])[:, None]
    return {"legs": legs, "points": centered @ axes.T, "variance": (singular ** 2 / np.sum(singular ** 2))[:3]}


def draw_legs(ax, essay: dict, parity: int, view: list[float]) -> None:
    """One cube of panel (c): the first legs of the pairs (parity 0: legs 1, 3, 5, 7) or the second (parity 1:
    legs 2, 4, 6, 8), each as a path through its fragments, in the projection and limits shared by both cubes."""
    legs, points = essay["legs"], essay["points"]
    bounds = np.cumsum([0] + [len(leg) for leg in legs])
    for index in range(parity, len(legs), 2):
        coords = points[bounds[index]:bounds[index + 1]]
        colors = [shade(PAIR_COLORS[index // 2], k / max(1, len(coords) - 1)) for k in range(len(coords))]
        for k in range(len(coords) - 1):
            ax.plot(*coords[k:k + 2].T, color=colors[k + 1], linewidth=1.8, solid_capstyle="round")
        ax.scatter(*coords.T, c=colors, s=26, edgecolors="white", linewidths=0.5, depthshade=False, zorder=3)
        ax.text(*coords[0], f" {index + 1}", fontsize=15, color=colors[-1], va="bottom")
    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.set_zlim(points[:, 2].min(), points[:, 2].max())
    ax.set_proj_type("ortho")
    ax.view_init(*view)
    ax.set_box_aspect(np.ptp(points, axis=0))
    for axis, name, share in zip((ax.xaxis, ax.yaxis, ax.zaxis), ("PC1", "PC2", "PC3"), essay["variance"]):
        axis.set_pane_color((0.975, 0.975, 0.975, 1.0))
        axis.pane.set_edgecolor("#d9d9d9")
        axis._axinfo["grid"].update(color="#e3e3e3", linewidth=0.6)
        # PCA coordinates have no meaningful units, so ticks carry no labels.
        axis.set_ticklabels([])
        axis._axinfo["tick"].update(inward_factor=0, outward_factor=0)
        axis.set_label_text(f"{name} ({share * 100:.1f}%)", fontsize=8, color=MUTED)
        axis.labelpad = -12


def draw_legs_flat(ax, essay: dict, parity: int) -> None:
    """One half of panel (c) in two dimensions: the first two principal components of the same projection, with
    the limits shared by both halves."""
    legs, points = essay["legs"], essay["points"][:, :2]
    bounds = np.cumsum([0] + [len(leg) for leg in legs])
    for index in range(parity, len(legs), 2):
        coords = points[bounds[index]:bounds[index + 1]]
        colors = [shade(PAIR_COLORS[index // 2], k / max(1, len(coords) - 1)) for k in range(len(coords))]
        for k in range(len(coords) - 1):
            ax.plot(*coords[k:k + 2].T, color=colors[k + 1], linewidth=1.8, solid_capstyle="round", zorder=2)
        ax.scatter(*coords.T, c=colors, s=30, edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate(str(index + 1), coords[0], xytext=(5, 4), textcoords="offset points", fontsize=15,
                    color=colors[-1], zorder=4)
    pad = 0.06 * np.ptp(points, axis=0)
    ax.set_xlim(points[:, 0].min() - pad[0], points[:, 0].max() + pad[0])
    ax.set_ylim(points[:, 1].min() - pad[1], points[:, 1].max() + pad[1])
    ax.set_aspect("equal")
    # PCA coordinates have no meaningful units, so the axes carry no ticks.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"PC1 ({essay['variance'][0] * 100:.1f}%)", fontsize=8, color=MUTED)
    ax.set_ylabel(f"PC2 ({essay['variance'][1] * 100:.1f}%)", fontsize=8, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbdbd")


def draw_figure(conditions: dict[str, list[dict]], summary: dict, labels: dict, highlight: str, essay: dict | None,
                args, output: Path) -> None:
    """Panels (a) and (b), and panel (c) if an essay projection is given."""
    highlight_label, highlight_id = highlight.rsplit("=", 1)
    real_labels, prose_labels = labels["real"], labels["prose"]
    layers = len(next(iter(conditions.values()))[0]["alignment"])
    L = args.figure_layer
    if essay is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    else:
        flat = args.essay_dims == 2
        fig = plt.figure(figsize=(14, 9.6 if flat else 10.6))
        outer = fig.add_gridspec(2, 1, height_ratios=[4.6, 5.0 if flat else 5.6], hspace=0.2 if flat else 0.22,
                                 left=0.05, right=0.99, top=0.97, bottom=0.02)
        top = outer[0].subgridspec(1, 2, width_ratios=[1.3, 1], wspace=0.12)
        axes = [fig.add_subplot(top[0]), fig.add_subplot(top[1])]
        bottom = outer[1].subgridspec(1, 3, width_ratios=[1, 1, 0.46], wspace=0.2 if flat else 0.02)
        ax_odd = fig.add_subplot(bottom[0], projection=None if flat else "3d")
        ax_even = fig.add_subplot(bottom[1], projection=None if flat else "3d")
    for label in conditions:
        axes[0].plot(range(layers), summary[label]["difference"], color=COLORS.get(label, "0.5"), linewidth=1.8,
                     label=f"{label} ({summary[label]['texts']} essays, {summary[label]['pairs']} pairs)" if args.dots == "texts"
                     else f"{label} ({summary[label]['pairs']} pairs)")
    axes[0].plot(range(layers), summary[highlight_label]["difference"], color=HIGHLIGHT_COLOR, linewidth=1.2,
                 linestyle="--", label=f"{highlight_label}, within {real_labels[0].split(' (')[0]} ({summary[highlight_label]['pairs']} pairs)")
    axes[0].axhline(0, color="0.6", linewidth=0.6)
    axes[0].set_xlabel("GujiBERT layer")
    axes[0].set_ylabel("Alignment minus random-pair baseline")
    axes[0].set_title("(a) Real leg pairs, controls and pseudo-leg pairs, by layer", fontsize=10, loc="left")
    # Two columns in the empty band between the prose line and the aligned conditions.
    axes[0].legend(frameon=False, fontsize=10, loc="center", bbox_to_anchor=(0.5, 0.42), ncol=2)
    for position, label in enumerate(conditions):
        records = conditions[label]
        text_level = summary[label]["text_level_at_figure_layer"]
        if args.dots == "texts":
            means = text_means(records, L)
            texts, values = list(means), np.array(list(means.values()))
            low, high = text_level["text_mean_ci95"]
        else:
            texts = [r["text"] for r in records]
            values = np.array([r["alignment"][L] - r["baseline"][L] for r in records])
            low, high = text_level["pair_mean_ci95"]
        jitter = np.random.default_rng(position).uniform(-0.2, 0.2, len(values))
        marked = np.array([t == highlight_id for t in texts]) if label == real_labels[0] else np.zeros(len(values), bool)
        small = label in prose_labels and len(values) > 1000
        axes[1].scatter(position + jitter[~marked], values[~marked], s=4 if small else (14 if args.dots == "texts" else 9),
                        color=COLORS.get(label, "0.5"), alpha=0.25 if small else 0.6, linewidths=0, zorder=2)
        axes[1].scatter(position + jitter[marked], values[marked], s=46, color=HIGHLIGHT_COLOR,
                        edgecolors="white", linewidths=0.8, zorder=4, label=highlight_label if marked.any() else None)
        axes[1].hlines(values.mean(), position - 0.3, position + 0.3, color="black", linewidth=1.5, zorder=3)
        axes[1].vlines(position + 0.34, low, high, color="black", linewidth=1.2, zorder=3)
    axes[1].axhline(0, color="0.6", linewidth=0.6)
    axes[1].set_xticks(range(len(conditions)))
    # Long labels wrap at their first space or parenthesis so that neighbouring labels do not collide.
    axes[1].set_xticklabels([(label.replace(" (", "\n(") if " (" in label else label.replace(" ", "\n", 1))
                             + (f"\n{summary[label]['texts']} essays" if args.dots == "texts" else f"\n{summary[label]['pairs']} pairs")
                             for label in conditions], fontsize=9)
    axes[1].set_title(f"(b) Every essay at layer {L} (mean of its pairs); bar = mean, whisker = 95% CI" if args.dots == "texts"
                      else f"(b) Every pair at layer {L}; bar = mean, whisker = 95% CI over texts", fontsize=10, loc="left")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    if essay is None:
        fig.tight_layout()
    else:
        for ax, parity in ((ax_odd, 0), (ax_even, 1)):
            if flat:
                draw_legs_flat(ax, essay, parity)
            else:
                draw_legs(ax, essay, parity, args.essay_view)
        ax_odd.set_title(f"(c) {highlight_label}'s essay at layer {L}, one PCA: legs 1, 3, 5, 7", fontsize=10, loc="left")
        ax_even.set_title("Legs 2, 4, 6, 8, same projection", fontsize=10, loc="left")
        handles = [Line2D([], [], color=color, linewidth=2.6) for color in PAIR_COLORS]
        names = [f"Legs {2 * p + 1} and {2 * p + 2}: {name}" for p, name in enumerate(PAIR_NAMES)]
        ax_even.legend(handles, names, frameon=False, fontsize=10, loc="center left",
                       bbox_to_anchor=(1.04 if flat else 1.0, 0.5),
                       title="Light to dark: first to last clause", title_fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def replot(args, model, tokenizer, device) -> None:
    """Redraw the figure of an earlier run from its summary.json and pairs.json."""
    output_dir = Path(args.output_dir)
    saved = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    pairs = json.loads((output_dir / "pairs.json").read_text(encoding="utf-8"))
    conditions = {label: [{**r, "alignment": np.array(r["alignment"]), "baseline": np.array(r["baseline"])} for r in records]
                  for label, records in pairs.items()}
    summary = saved["summary"]
    # Runs before labels were saved: the first condition is the corpus, prose conditions have length statistics.
    labels = saved.get("labels") or {"real": [next(iter(conditions))],
                                     "prose": [l for l in conditions if "length_gap_correlation" in summary[l]]}
    highlight = saved["highlight"]
    args.figure_layer = saved["figure_layer"]
    for label in args.drop:
        conditions.pop(label)
    # A substituted condition takes the place of the saved one, with its summary computed from its own pairs
    # (e.g. the enciphered essays read by the cipher model, scripts/cipher_bert/align_essays.py).
    np_rng = np.random.default_rng(args.seed)
    for spec in args.substitute:
        label, path = spec.split("=", 1)
        records = [{**r, "alignment": np.array(r["alignment"]), "baseline": np.array(r["baseline"])}
                   for r in json.loads(Path(path).read_text(encoding="utf-8"))[label]]
        conditions[label] = records
        L = args.figure_layer
        diff = np.array([r["alignment"] - r["baseline"] for r in records])
        means = np.array(list(text_means(records, L).values()))
        text_boot = means[np_rng.integers(0, len(means), size=(args.bootstrap, len(means)))].mean(axis=1)
        summary[label] = {"pairs": len(records), "texts": len(means), "difference": diff.mean(axis=0).tolist(),
                          "text_level_at_figure_layer": {
                              "text_mean_ci95": np.quantile(text_boot, [0.025, 0.975]).tolist(),
                              "pair_mean_ci95": np.quantile(bootstrap_texts(records, L, args.bootstrap, np_rng),
                                                            [0.025, 0.975]).tolist()}}
    if args.order:
        conditions = {label: conditions[label] for label in args.order}
    essay = None
    if not args.no_essay_panel:
        corpus_path = args.corpus[0].rsplit("=", 1)[1]
        essay = essay_projection(corpus_path, highlight.rsplit("=", 1)[1], args.figure_layer, args, model, tokenizer, device)
    draw_figure(conditions, summary, labels, highlight, essay, args, output_dir / args.figure_name)
    print(f"Redrawn: {output_dir / args.figure_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", nargs="*", default=["欽定四書文 (100 essays)=data/4_eight_legged_essays/qinding_legs.json"],
                        help="label=path pairs of eight-legged essays with marked parallel pairs")
    parser.add_argument("--highlight", default="Chen Zilong=346",
                        help="label=essay id of the first corpus whose pairs are reported and drawn separately")
    parser.add_argument("--controls", nargs="*", default=["Composed (100 essays)=data/4_eight_legged_essays/qinding_controls/composed.json",
                                                           "Nonsense (100 essays)=data/4_eight_legged_essays/qinding_controls/nonsense.json"],
                        help="label=path pairs of control essays built on the first corpus, matched by essay id")
    parser.add_argument("--min-fragments", type=int, default=3,
                        help="Pairs with fewer fragments per member are left out")
    parser.add_argument("--prose", nargs="*", default=["古文觀止=data/4_eight_legged_essays/pseudo_legs/guwen.json"],
                        help="label=path pairs of non-parallel prose, either pieces (pseudo-leg pairs are sampled "
                             "from them) or pseudo-essays matched to the corpus (make_pseudo_essays.py); the first "
                             "is the reference for comparisons")
    parser.add_argument("--baseline-pool", default="data/4_eight_legged_essays/pseudo_legs/guwen.json",
                        help="Prose pieces whose pseudo-leg pairs are the random pairs of the baseline")
    parser.add_argument("--pairs-per-n", type=int, default=300, help="Pseudo-leg pairs sampled per corpus and n")
    parser.add_argument("--min-chars", type=int, help="Default: the shortest real fragment")
    parser.add_argument("--max-chars", type=int, help="Default: the longest real fragment")
    parser.add_argument("--random-pairs", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=10000, help="Resamples of texts for confidence intervals")
    parser.add_argument("--figure-layer", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pairs-per-batch", type=int, default=16)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dots", choices=["pairs", "texts"], default="pairs",
                        help="Panel (b): one dot per pair, or one per text (the mean of its pairs)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--essay-dims", type=int, choices=[2, 3], default=2,
                        help="Panel (c): the first two principal components in two panels, or three in two cubes")
    parser.add_argument("--essay-view", type=float, nargs=2, default=[45, 40], metavar=("ELEV", "AZIM"),
                        help="Panel (c) with --essay-dims 3: elevation and azimuth of both cubes")
    parser.add_argument("--no-essay-panel", action="store_true", help="Draw panels (a) and (b) only")
    parser.add_argument("--replot", action="store_true",
                        help="Redraw the figure from summary.json and pairs.json in --output-dir without scoring again")
    parser.add_argument("--drop", nargs="*", default=[], help="With --replot: conditions left out of the figure")
    parser.add_argument("--substitute", nargs="*", default=[],
                        help="With --replot: label=path of a pairs.json whose condition `label` replaces the saved one")
    parser.add_argument("--figure-name", default="pseudo_legs.png", help="With --replot: file name of the figure")
    parser.add_argument("--order", nargs="*", default=[], help="With --replot: the conditions in the order drawn")
    parser.add_argument("--output-dir", default="results/baguwen_pseudo_legs")
    args = parser.parse_args()

    load_fonts()
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    if args.replot:
        replot(args, model, tokenizer, device)
        return
    rng = random.Random(args.seed)

    conditions: dict[str, list[dict]] = {}
    real_labels, control_labels = [], []
    for specs, labels in ((args.corpus, real_labels), (args.controls, control_labels)):
        for spec in specs:
            label, path = spec.rsplit("=", 1)
            conditions[label] = pair_steps(load_corpus(path, args.min_fragments), args, model, tokenizer, device)
            labels.append(label)
    highlight_label, highlight_id = args.highlight.rsplit("=", 1)
    highlighted = [r for r in conditions[real_labels[0]] if r["text"] == highlight_id]
    real = [f for records in conditions.values() for r in records for f in r["a"] + r["b"]]
    args.min_chars = args.min_chars or min(len(f) for f in real)
    args.max_chars = args.max_chars or max(len(f) for f in real)
    print(f"Fragments of {args.min_chars} to {args.max_chars} characters")
    sizes = sorted({len(r["a"]) for label in real_labels for r in conditions[label]})
    def sample_windows(path: str, label: str) -> list[tuple[list[str], list[str], str, None]]:
        pieces = json.loads(Path(path).read_text(encoding="utf-8"))["pieces"]
        sampled = []
        for n in sizes:
            windows = prose_windows(pieces, n, args.min_chars, args.max_chars)
            chosen = rng.sample(windows, min(args.pairs_per_n, len(windows)))
            print(f"{label}, n={n}: {len(windows)} windows available, {len(chosen)} used")
            sampled.extend(chosen)
        return sampled

    prose_labels = []
    for spec in args.prose:
        label, path = spec.rsplit("=", 1)
        matched = "essays" in json.loads(Path(path).read_text(encoding="utf-8"))
        pairs = load_corpus(path, args.min_fragments) if matched else sample_windows(path, label)
        conditions[label] = pair_steps(pairs, args, model, tokenizer, device)
        prose_labels.append(label)
    reference_label = prose_labels[0]
    # Random pairs for the baseline: pseudo-leg pairs of every n, apart from the conditions shown.
    pool = pair_steps(sample_windows(args.baseline_pool, "baseline pool"), args, model, tokenizer, device)
    reference = {n: [r for r in pool if len(r["a"]) == n] for n in sizes}
    for records in [*conditions.values(), pool]:
        score(records, reference, args.random_pairs, rng)

    layers = len(conditions[reference_label][0]["alignment"])
    reference_diff = np.array([r["alignment"] - r["baseline"] for r in conditions[reference_label]])
    summary = {}
    # The highlighted essay's pairs stay in their corpus; its entry only reports them on their own.
    for label, records in [*conditions.items(), (highlight_label, highlighted)]:
        aligned = np.array([r["alignment"] for r in records])
        diff = np.array([r["alignment"] - r["baseline"] for r in records])
        entry = {"pairs": len(records), "texts": len({r["text"] for r in records}),
                 "pairs_by_n": {str(n): sum(len(r["a"]) == n for r in records) for n in sizes},
                 "mean_length_gap": float(np.mean([r["length_gap"] for r in records])),
                 "mean_shared_characters": float(np.mean([r["shared"] for r in records])),
                 "alignment": aligned.mean(axis=0).tolist(), "baseline": np.array([r["baseline"] for r in records]).mean(axis=0).tolist(),
                 "difference": diff.mean(axis=0).tolist(), "difference_sd": diff.std(axis=0).tolist()}
        if label != reference_label and len(records) > 1:
            entry["vs_reference_mann_whitney_p"] = [float(mannwhitneyu(diff[:, L], reference_diff[:, L],
                                                                       alternative="greater").pvalue) for L in range(layers)]
            entry["vs_reference_cohens_d"] = [cohens_d(diff[:, L], reference_diff[:, L]) for L in range(layers)]
        summary[label] = entry
    if highlight_label in conditions:
        raise SystemExit(f"--highlight label {highlight_label!r} collides with a condition")

    # Text-level comparisons at the figure layer: each essay or prose piece counts once.
    L = args.figure_layer
    np_rng = np.random.default_rng(args.seed)
    reference_texts = np.array(list(text_means(conditions[reference_label], L).values()))
    reference_boot = bootstrap_texts(conditions[reference_label], L, args.bootstrap, np_rng)
    boots = {}
    for label, records in conditions.items():
        means = np.array(list(text_means(records, L).values()))
        boots[label] = bootstrap_texts(records, L, args.bootstrap, np_rng)
        text_boot = means[np_rng.integers(0, len(means), size=(args.bootstrap, len(means)))].mean(axis=1)
        entry = {"texts": len(means), "mean_of_text_means": float(means.mean()),
                 "text_mean_ci95": np.quantile(text_boot, [0.025, 0.975]).tolist(),
                 "pair_mean_ci95": np.quantile(boots[label], [0.025, 0.975]).tolist()}
        if label != reference_label:
            gap = boots[label] - reference_boot
            entry.update({"vs_reference_text_mann_whitney_p": float(mannwhitneyu(means, reference_texts, alternative="greater").pvalue),
                          "vs_reference_text_cohens_d": cohens_d(means, reference_texts),
                          "vs_reference_difference_ci95": np.quantile(gap, [0.025, 0.975]).tolist()})
        summary[label]["text_level_at_figure_layer"] = entry
    real_means = text_means(conditions[real_labels[0]], L)
    for label in control_labels:
        control_means = text_means(conditions[label], L)
        shared = sorted(set(real_means) & set(control_means))
        x, y = np.array([real_means[t] for t in shared]), np.array([control_means[t] for t in shared])
        summary[label]["text_level_at_figure_layer"]["vs_real_paired"] = {
            "essays": len(shared), "real_minus_control": float((x - y).mean()),
            "wilcoxon_p": float(wilcoxon(x, y).pvalue), "real_higher": int((x > y).sum())}
    # Where each real pair falls among reference pseudo-leg pairs with the same n.
    for label in real_labels:
        summary[label]["percentile_in_reference"] = [
            {"n": len(r["a"]), "essay": r["text"],
             "layer_percentiles": [float(np.mean(np.array([o["alignment"][L] - o["baseline"][L] for o in reference[len(r["a"])]])
                                                 < r["alignment"][L] - r["baseline"][L])) for L in range(layers)]}
            for r in conditions[label]]
    L = args.figure_layer
    # By position of the pair in its essay (起股, 中股, 後股, 束股 in regular essays), as couplet positions in poems.
    for label in real_labels + control_labels:
        positions = sorted({r["position"] for r in conditions[label]})
        summary[label]["by_position_at_figure_layer"] = [
            {"position": p, "pairs": sum(r["position"] == p for r in conditions[label]),
             "difference": float(np.mean([r["alignment"][L] - r["baseline"][L] for r in conditions[label] if r["position"] == p]))}
            for p in positions]
    # Alignment at equal levels of verbatim overlap between corresponding fragments.
    bins = [0, 0.05, 0.15, 0.25, 0.4, 1.01]
    for label, records in conditions.items():
        shares = np.array([r["shared"] for r in records])
        diffs = np.array([r["alignment"][L] - r["baseline"][L] for r in records])
        summary[label]["by_shared_characters_at_figure_layer"] = [
            {"from": low, "to": high, "pairs": int(((shares >= low) & (shares < high)).sum()),
             "difference": float(diffs[(shares >= low) & (shares < high)].mean()) if ((shares >= low) & (shares < high)).any() else None}
            for low, high in zip(bins[:-1], bins[1:])]
    # Does a pseudo-leg pair align more when its corresponding fragments have similar lengths?
    for label in prose_labels:
        records = conditions[label]
        gaps = np.array([r["length_gap"] for r in records])
        diffs = np.array([r["alignment"][L] - r["baseline"][L] for r in records])
        summary[label]["length_gap_correlation"] = float(np.corrcoef(gaps, diffs)[0, 1])
        close = gaps <= np.quantile(gaps, 0.1)
        summary[label]["closest_length_decile_difference"] = float(diffs[close].mean())

    print("layer  " + "  ".join(f"{label:>22}" for label in conditions))
    print("       " + "  ".join(f"{'aligned / base / diff':>22}" for _ in conditions))
    for layer in range(layers):
        print(f"{layer:5d}  " + "  ".join(f"{summary[l]['alignment'][layer]:7.3f}{summary[l]['baseline'][layer]:7.3f}"
                                          f"{summary[l]['difference'][layer]:8.3f}" for l in conditions))
    for label, entry in summary.items():
        extra = ""
        if "vs_reference_cohens_d" in entry:
            extra = (f" d vs {reference_label} at layer {L}: {entry['vs_reference_cohens_d'][L]:.2f}"
                     f" (p={entry['vs_reference_mann_whitney_p'][L]:.1e})")
        print(f"{label}: {entry['pairs']} pairs in {entry['texts']} texts, mean length gap {entry['mean_length_gap']:.2f}, "
              f"shared characters {entry['mean_shared_characters']:.2f}{extra}")
    print(f"Layer {L}, texts as the unit:")
    for label in conditions:
        t = summary[label]["text_level_at_figure_layer"]
        line = f"  {label}: {t['texts']} texts, pair mean 95% CI [{t['pair_mean_ci95'][0]:.3f}, {t['pair_mean_ci95'][1]:.3f}]"
        if "vs_reference_text_cohens_d" in t:
            line += (f"; vs {reference_label}: d={t['vs_reference_text_cohens_d']:.2f}, p={t['vs_reference_text_mann_whitney_p']:.1e}, "
                     f"difference CI [{t['vs_reference_difference_ci95'][0]:.3f}, {t['vs_reference_difference_ci95'][1]:.3f}]")
        if "vs_real_paired" in t:
            v = t["vs_real_paired"]
            line += (f"; real minus control {v['real_minus_control']:.3f} over {v['essays']} essays "
                     f"(real higher in {v['real_higher']}, Wilcoxon p={v['wilcoxon_p']:.1e})")
        print(line)
    for label in prose_labels:
        print(f"{label}: corr(length gap, difference) {summary[label]['length_gap_correlation']:.2f}; "
              f"closest-length decile difference {summary[label]['closest_length_decile_difference']:.3f}")
    print(f"Layer {L} difference by share of shared characters:")
    for label in conditions:
        print(f"  {label}: " + "  ".join(
            f"[{x['from']:.2f},{x['to']:.2f}) " + (f"{x['difference']:.3f} ({x['pairs']})" if x["difference"] is not None else "- (0)")
            for x in summary[label]["by_shared_characters_at_figure_layer"]))
    for label in real_labels + control_labels:
        print(f"  {label} by pair position: " + "  ".join(f"{x['position']}: {x['difference']:.3f} ({x['pairs']})"
                                                       for x in summary[label]["by_position_at_figure_layer"]))
    print(f"{highlight_label}: pairs at layer {L}:", [round(float(r["alignment"][L] - r["baseline"][L]), 3) for r in highlighted])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {"real": real_labels, "controls": control_labels, "prose": prose_labels}
    essay = None if args.no_essay_panel else essay_projection(args.corpus[0].rsplit("=", 1)[1], highlight_id, L,
                                                              args, model, tokenizer, device)
    draw_figure(conditions, summary, labels, args.highlight, essay, args, output_dir / "pseudo_legs.png")

    pairs_out = {label: [{"text": r["text"], "position": r["position"], "a": r["a"], "b": r["b"], "shared": round(r["shared"], 3), "alignment": np.round(r["alignment"], 4).tolist(),
                          "baseline": np.round(r["baseline"], 4).tolist(), "length_gap": r["length_gap"]}
                         for r in records] for label, records in conditions.items()}
    (output_dir / "summary.json").write_text(json.dumps({
        "model": args.model, "reference": reference_label, "highlight": args.highlight, "fragment_sizes": sizes,
        "min_chars": args.min_chars, "max_chars": args.max_chars, "random_pairs": args.random_pairs,
        "bootstrap": args.bootstrap, "figure_layer": L, "seed": args.seed, "labels": labels, "dots": args.dots,
        "essay_projection": None if essay is None else {"layer": L, "dims": args.essay_dims,
                                                        "view": args.essay_view if args.essay_dims == 3 else None,
                                                        "variance": np.round(essay["variance"], 4).tolist()},
        "summary": summary,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "pairs.json").write_text(json.dumps(pairs_out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
