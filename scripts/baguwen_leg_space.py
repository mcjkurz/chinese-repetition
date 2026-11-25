"""The essay as a figure in embedding space: each section is the mean vector of its characters.

The whole essay is encoded in context once; at every layer each section (破題,
承題, 起講, legs 1-8, 大結) becomes the mean of its characters' embeddings,
punctuation left out, after the embeddings are centred on the essay's mean.

A mean over n characters keeps noise in proportion to 1/n, so a short section
(legs 3-4) would sit farther from the centre than a long one only because it
is short. In the corrected geometry the squared length of each
section mean is replaced by the mean dot product over pairs of distinct
characters in the section, which estimates the squared length of the section's
true mean without that noise term. Neighbouring characters share context and
are not independent, so pairs fewer than --min-gap positions apart are left out
of that estimate (with --min-gap 3 the scrambled essays, whose sections differ
only in length, show no relation between length and distance from the centre); dot products between different sections are
left as they are. The resulting Gram matrix is double-centred and its leading
three eigenvectors give the 3D coordinates (classical MDS). Both this geometry
and the plain one (ordinary PCA of the uncorrected section means) are computed,
and the page switches between them. Two kinds of connection are drawn:

- the path: consecutive sections in the order of the essay (or, on the page,
  the first members of the pairs, legs 1-3-5-7, and the second, legs 2-4-6-8);
- the pairs: each leg joined to its partner (1-2, 3-4, 5-6, 7-8), as the two
  lines of a couplet.

Statistics are computed in the full space (not the projection), from the same
Gram matrix:

- partner rank: for each leg, the rank of its partner among the other eleven
  sections by cosine similarity (1 = nearest; about 6 by chance);
- arrow alignment: mean cosine between the pair arrows (leg 2 - leg 1, leg 4 -
  leg 3, ...); positive when the pairs turn in a shared direction, as
  corresponding words of a couplet do. The cosine of every two arrows is kept
  as well, since lines that look parallel in the projection need not be in the
  full space;
- length effect: Spearman correlation between a section's number of characters
  and its distance from the centre (near 0 when length does not place sections).

The same statistics are computed for the control essays. Writes a static 3D
figure for each --figure-layers entry, an interactive HTML page with a layer
slider, and a JSON file with coordinates and statistics.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qhchina import load_fonts
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

from common import select_device
from lm_scoring_utils import PUNCTUATION
from make_baguwen_figure import section_bounds
from self_similarity_matrix import DEFAULT_MODEL, all_layer_character_embeddings

SECTIONS = ["破題", "承題", "起講", "Leg 1", "Leg 2", "Leg 3", "Leg 4", "Leg 5", "Leg 6", "Leg 7", "Leg 8", "大結"]
PAIRS = [(3, 4), (5, 6), (7, 8), (9, 10)]
PAIR_NAMES = ["起股 (legs 1-2)", "中股 (legs 3-4)", "後股 (legs 5-6)", "束股 (legs 7-8)"]
PAIR_COLORS = ["#1f4e8c", "#c2571a", "#2a9d8f", "#8e44ad"]
OTHER_COLOR = "#6b6b6b"
GEOMETRIES = ("corrected", "mean")


def section_grams(raw_text: str, geometry: str, min_gap: int, model, tokenizer, device) -> tuple[np.ndarray, list[int], list[str]]:
    """Double-centred Gram matrix of the section means at every layer, shape (layers, sections, sections),
    with the sections' character counts and texts."""
    text, bounds = section_bounds(raw_text)
    chars, hidden = all_layer_character_embeddings(text, model, tokenizer, device)
    if len(chars) != len(text):
        raise ValueError("Tokenization does not align with characters")
    if len(bounds) - 1 != len(SECTIONS):
        raise ValueError(f"Expected {len(SECTIONS)} sections, found {len(bounds) - 1}")
    members = [[i for i in range(start, end) if text[i] not in PUNCTUATION] for start, end in zip(bounds[:-1], bounds[1:])]
    hidden = hidden - hidden[:, sorted(i for indices in members for i in indices)].mean(axis=1, keepdims=True)
    means = np.stack([hidden[:, indices].mean(axis=1) for indices in members], axis=1)
    grams = means @ means.transpose(0, 2, 1)
    if geometry == "corrected":
        for k, indices in enumerate(members):
            positions = np.array(indices)
            far = np.abs(positions[:, None] - positions[None, :]) >= min_gap
            products = hidden[:, indices] @ hidden[:, indices].transpose(0, 2, 1)
            grams[:, k, k] = (products * far).sum(axis=(1, 2)) / far.sum()
    centring = np.eye(len(members)) - 1 / len(members)
    return centring @ grams @ centring, [len(indices) for indices in members], \
        [text[start:end] for start, end in zip(bounds[:-1], bounds[1:])]


def pair_statistics(gram: np.ndarray, lengths: list[int]) -> dict[str, float]:
    """Partner rank, arrow alignment and length effect of one layer's Gram matrix, in the full space."""
    norms = np.sqrt(np.clip(np.diag(gram), 1e-12, None))
    similarity = gram / np.outer(norms, norms)
    ranks = []
    for a, b in PAIRS:
        for leg, partner in ((a, b), (b, a)):
            others = np.delete(np.arange(len(gram)), leg)
            order = others[np.argsort(-similarity[leg, others])]
            ranks.append(int(np.where(order == partner)[0][0]) + 1)
    # Dot products of the arrows m_b - m_a, from the Gram matrix alone.
    arrows = np.array([[gram[b, d] - gram[b, c] - gram[a, d] + gram[a, c] for c, d in PAIRS] for a, b in PAIRS])
    # With the correction an arrow between two near-identical legs can get a squared length <= 0; its
    # direction is then undefined and its cosines are left out.
    arrow_norms = np.sqrt(np.where(np.diag(arrows) > 0, np.diag(arrows), np.nan))
    arrows = np.clip(arrows / np.outer(arrow_norms, arrow_norms), -1, 1)
    return {
        "partner_mean_rank": float(np.mean(ranks)),
        "partner_nearest_share": float(np.mean(np.array(ranks) == 1)),
        "arrow_alignment": float(np.nanmean([arrows[i, j] for i, j in combinations(range(len(PAIRS)), 2)])),
        "arrow_cosines": [[None if np.isnan(value) else float(value) for value in row] for row in arrows],
        "length_effect": float(spearmanr(lengths, norms).correlation),
    }


def project(grams: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """3D coordinates from the leading eigenvectors of each layer's Gram matrix, signs matched across layers,
    and the share of the total variance on each axis."""
    coordinates, shares, previous = [], [], None
    for gram in grams:
        values, vectors = np.linalg.eigh(gram)
        order = np.argsort(values)[::-1][:3]
        points = vectors[:, order] * np.sqrt(np.clip(values[order], 0, None))
        if previous is not None:
            points *= np.sign(np.sum(points * previous, axis=0)) + (np.sum(points * previous, axis=0) == 0)
        coordinates.append(points)
        shares.append(values[order] / np.trace(gram))
        previous = points
    return np.array(coordinates), np.array(shares)


def plot_static(points: np.ndarray, shares: np.ndarray, layer: int, geometry: str, output: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect(None, zoom=0.85)
    # The path through the sections in order, light at the start and dark at the end.
    shades = plt.get_cmap("Greys")(np.linspace(0.3, 0.85, len(points) - 1))
    for start, shade in enumerate(shades):
        ax.plot(*points[start:start + 2].T, color=shade, linewidth=1.4)
    ax.plot([], [], color=OTHER_COLOR, linewidth=1.4, label="Order of the essay (light to dark)")
    for (a, b), name, color in zip(PAIRS, PAIR_NAMES, PAIR_COLORS):
        ax.plot(*points[[a, b]].T, color=color, linewidth=3, label=name)
    paired = {index: color for (a, b), color in zip(PAIRS, PAIR_COLORS) for index in (a, b)}
    for index, (point, name) in enumerate(zip(points, SECTIONS)):
        color = paired.get(index, OTHER_COLOR)
        ax.scatter(*point, color=color, s=40, depthshade=False)
        ax.text(*point, f"  {name}", fontsize=9, color=color)
    for axis, share, setter in zip("123", shares, (ax.set_xlabel, ax.set_ylabel, ax.set_zlabel)):
        setter(f"Axis {axis} ({share:.0%})")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    kind = "noise-corrected section means" if geometry == "corrected" else "section means"
    ax.set_title(f"Chen Zilong: {kind}, GujiBERT layer {layer}", fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def write_html(payload: dict, output: Path) -> None:
    output.write_text(HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="data/4_eight_legged_essays/baguwen_chen_zilong.txt")
    parser.add_argument(
        "--controls",
        nargs="*",
        default=["Composed=data/4_eight_legged_essays/baguwen_controls/sensical.json",
                 "Nonsense=data/4_eight_legged_essays/baguwen_controls/nonsense.json",
                 "Scramble=data/4_eight_legged_essays/baguwen_controls/scrambled.json"],
        help="label=path pairs; each JSON has a 'passages' list of {id, text}",
    )
    parser.add_argument("--figure-layers", type=int, nargs="+", default=[8, 12],
                        help="Layers drawn as static figures; the first is the page's starting layer")
    parser.add_argument("--min-gap", type=int, default=3,
                        help="Character pairs closer than this are left out of the corrected section lengths")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-dir", default="results/baguwen_leg_space")
    args = parser.parse_args()

    load_fonts()
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    output_dir = Path(args.output_dir)

    payload = {
        "model": args.model, "original": args.original, "min_gap": args.min_gap, "sections": SECTIONS, "pairs": PAIRS,
        "pair_names": PAIR_NAMES, "pair_colors": PAIR_COLORS, "other_color": OTHER_COLOR,
        "default_layer": args.figure_layers[0], "geometries": {},
    }
    for geometry in GEOMETRIES:
        grams, lengths, texts = section_grams(Path(args.original).read_text(encoding="utf-8"), geometry, args.min_gap,
                                             model, tokenizer, device)
        coordinates, shares = project(grams)
        original_stats = [pair_statistics(gram, lengths) for gram in grams]

        control_stats = {}
        for spec in args.controls:
            label, path = spec.rsplit("=", 1)
            passages = json.loads(Path(path).read_text(encoding="utf-8"))["passages"]
            per_essay = []
            for passage in passages:
                passage_grams, passage_lengths, _ = section_grams(passage["text"], geometry, args.min_gap,
                                                                  model, tokenizer, device)
                per_essay.append([pair_statistics(gram, passage_lengths) for gram in passage_grams])
            control_stats[label] = [{key: np.nanmean(np.array([essay[layer][key] for essay in per_essay], dtype=float),
                                                     axis=0).tolist()
                                     for key in original_stats[0]} for layer in range(len(grams))]

        print(f"{geometry} geometry")
        print("layer  " + "  ".join(f"{name:>26}" for name in ["Chen", *control_stats]))
        print("       " + "  ".join(f"{'rank / arrows / length':>26}" for _ in range(1 + len(control_stats))))
        for layer in range(len(grams)):
            rows = [original_stats[layer], *(stats[layer] for stats in control_stats.values())]
            print(f"{layer:5d}  " + "  ".join(
                f"{r['partner_mean_rank']:8.2f} {r['arrow_alignment']:8.2f} {r['length_effect']:8.2f}"
                for r in rows))

        stem = "leg_space" if geometry == "corrected" else "leg_space_mean"
        for layer in args.figure_layers:
            plot_static(coordinates[layer], shares[layer], layer, geometry,
                        output_dir / f"{stem}_layer{layer:02d}.png", args.dpi)
        payload.update({"section_texts": texts, "section_lengths": lengths})
        payload["geometries"][geometry] = {
            "coordinates": coordinates.round(4).tolist(), "variance_shares": shares.round(4).tolist(),
            "statistics": {"Chen Zilong": original_stats, **control_stats},
        }

    (output_dir / "leg_space.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    write_html(payload, output_dir / "leg_space.html")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Leg space</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
  :root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6b6b6b; --line: #e3e3e6; --panel: #f6f6f8; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) { --bg: #16171a; --fg: #ececf0; --muted: #a0a0a8; --line: #2c2d33; --panel: #1f2025; }
  }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 15px/1.5 -apple-system, "Segoe UI", "PingFang TC", "Noto Sans TC", sans-serif; }
  main { max-width: 1200px; margin: 0 auto; padding: 20px 16px 40px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  p.lede { color: var(--muted); margin: 0 0 16px; max-width: 75ch; }
  .layout { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 20px; }
  @media (max-width: 860px) { .layout { grid-template-columns: 1fr; } }
  #plot { height: 640px; border: 1px solid var(--line); border-radius: 8px; }
  aside { display: flex; flex-direction: column; gap: 14px; }
  .card { background: var(--panel); border-radius: 8px; padding: 12px 14px; }
  .card h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 0 0 8px; }
  label { display: block; margin: 4px 0; }
  input[type=range] { width: 100%; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
  th, td { text-align: right; padding: 3px 4px; border-bottom: 1px solid var(--line); }
  th:first-child, td:first-child { text-align: left; }
  #section { font-size: 14px; min-height: 4em; }
  .note { color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<main>
  <h1>Chen Zilong's essay in GujiBERT space</h1>
  <p class="lede">Each point is the mean embedding of one section's characters, projected onto the three main
  axes of the twelve section means. <span id="geometry"></span> Thick segments join partner legs; the thin line follows
  the order of the essay, light at the start and dark at the end. Drag to rotate, click a legend entry to hide it, hover or click a point to read its text.</p>
  <div class="layout">
    <div id="plot"></div>
    <aside>
      <div class="card">
        <label>Geometry <select id="geometry-choice">
          <option value="corrected">Corrected for section length</option>
          <option value="mean">Plain section means</option>
        </select></label>
        <h2>Layer <span id="layer-value"></span></h2>
        <input id="layer" type="range" min="0" step="1">
        <label>Path <select id="path">
          <option value="essay">Order of the essay</option>
          <option value="strings">Odd legs (1-3-5-7) and even legs (2-4-6-8)</option>
          <option value="none">None</option>
        </select></label>
        <label><input id="pairs" type="checkbox" checked> Show the leg pairs</label>
        <div class="note" id="variance"></div>
      </div>
      <div class="card">
        <h2>Pairing in the full space</h2>
        <table id="stats"></table>
        <p class="note">Partner rank: rank of each leg's partner among the other eleven sections by cosine
        (1 = nearest, about 6 by chance). Nearest: share of legs whose partner is nearest. Arrows: mean cosine
        between the pair arrows (leg 2 − leg 1, leg 4 − leg 3, …). Length: rank correlation between a section's
        number of characters and its distance from the centre (near 0 = length does not place it). Controls are
        means over 50 essays.</p>
      </div>
      <div class="card">
        <h2>Are the pair lines parallel?</h2>
        <table id="arrows"></table>
        <p class="note">Cosine between two pair arrows (leg 2 − leg 1, leg 4 − leg 3, …): near +1 or −1 the two
        lines look parallel, near 0 they are perpendicular. 3D: in the projection shown. The other columns use the
        full space; controls are means over 50 essays.</p>
      </div>
      <div class="card">
        <h2>Section</h2>
        <div id="section">Hover over a point.</div>
      </div>
    </aside>
  </div>
</main>
<script>
const data = __DATA__;
const slider = document.getElementById("layer");
const choice = document.getElementById("geometry-choice");
const geo = () => data.geometries[choice.value];
slider.max = data.geometries.corrected.coordinates.length - 1;
slider.value = data.default_layer;
const paired = {};
data.pairs.forEach(([a, b], k) => { paired[a] = data.pair_colors[k]; paired[b] = data.pair_colors[k]; });
const colorOf = i => paired[i] || data.other_color;
const dark = () => matchMedia("(prefers-color-scheme: dark)").matches;

function traces(layer) {
  const p = geo().coordinates[layer];
  const col = k => p.map(row => row[k]);
  const out = [];
  const path = document.getElementById("path").value;
  if (path === "essay") {
    out.push({type: "scatter3d", mode: "lines", name: "Order of the essay", x: col(0), y: col(1), z: col(2),
              line: {color: p.map((_, i) => i), colorscale: [[0, "#cfcfd4"], [1, "#2f2f35"]], width: 5},
              hoverinfo: "skip"});
  } else if (path === "strings") {
    [["Legs 1-3-5-7", data.pairs.map(([a]) => a), "#2f2f35"],
     ["Legs 2-4-6-8", data.pairs.map(([, b]) => b), "#9a9aa2"]].forEach(([name, members, color]) => out.push({
      type: "scatter3d", mode: "lines", name, x: members.map(i => p[i][0]), y: members.map(i => p[i][1]),
      z: members.map(i => p[i][2]), line: {color, width: 5}, hoverinfo: "skip"}));
  }
  if (document.getElementById("pairs").checked) {
    data.pairs.forEach(([a, b], k) => out.push({
      type: "scatter3d", mode: "lines", name: data.pair_names[k], x: [p[a][0], p[b][0]], y: [p[a][1], p[b][1]],
      z: [p[a][2], p[b][2]], line: {color: data.pair_colors[k], width: 10}, hoverinfo: "skip"}));
  }
  out.push({type: "scatter3d", mode: "markers+text", name: "Sections", showlegend: false,
            x: col(0), y: col(1), z: col(2), text: data.sections, textposition: "top center",
            textfont: {color: data.sections.map((_, i) => colorOf(i)), size: 13},
            marker: {size: 6, color: data.sections.map((_, i) => colorOf(i))},
            customdata: data.section_texts.map(t => t.length > 40 ? t.slice(0, 40) + "…" : t),
            hovertemplate: "<b>%{text}</b><br>%{customdata}<extra></extra>"});
  return out;
}

function layout(layer) {
  const s = geo().variance_shares[layer];
  const fg = dark() ? "#ececf0" : "#1d1d1f", grid = dark() ? "#34353b" : "#e3e3e6";
  const axis = t => ({title: {text: t}, color: fg, gridcolor: grid, zerolinecolor: grid, showbackground: false});
  return {uirevision: "keep", paper_bgcolor: "rgba(0,0,0,0)", font: {color: fg}, margin: {l: 0, r: 0, t: 0, b: 0},
          legend: {x: 0, y: 1, bgcolor: "rgba(0,0,0,0)"},
          scene: {xaxis: axis(`Axis 1 (${Math.round(s[0] * 100)}%)`), yaxis: axis(`Axis 2 (${Math.round(s[1] * 100)}%)`),
                  zaxis: axis(`Axis 3 (${Math.round(s[2] * 100)}%)`), aspectmode: "cube"}};
}

function render() {
  const layer = +slider.value;
  document.getElementById("layer-value").textContent = layer;
  const s = geo().variance_shares[layer];
  document.getElementById("variance").textContent =
    `The three components hold ${Math.round((s[0] + s[1] + s[2]) * 100)}% of the variance of the section means.`;
  Plotly.react("plot", traces(layer), layout(layer), {responsive: true, displaylogo: false});
  const rows = Object.entries(geo().statistics).map(([name, st]) => {
    const r = st[layer];
    return `<tr><td>${name}</td><td>${r.partner_mean_rank.toFixed(2)}</td>` +
           `<td>${Math.round(r.partner_nearest_share * 100)}%</td><td>${r.arrow_alignment.toFixed(2)}</td>` +
           `<td>${r.length_effect.toFixed(2)}</td></tr>`;
  });
  document.getElementById("stats").innerHTML =
    "<tr><th></th><th>Rank</th><th>Nearest</th><th>Arrows</th><th>Length</th></tr>" + rows.join("");

  const p = geo().coordinates[layer];
  const arrows = data.pairs.map(([a, b]) => { const d = p[b].map((v, k) => v - p[a][k]);
                                              const n = Math.hypot(...d); return d.map(v => v / n); });
  const short = ["1–2", "3–4", "5–6", "7–8"];
  const columns = ["Chen Zilong", "Composed", "Scramble"];
  const arrowRows = [];
  for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) {
    const projected = arrows[i].reduce((sum, v, k) => sum + v * arrows[j][k], 0);
    const cells = columns.map(c => { const v = geo().statistics[c][layer].arrow_cosines[i][j];
                                     return v === null ? "–" : v.toFixed(2); });
    arrowRows.push(`<tr><td>${short[i]} / ${short[j]}</td><td>${projected.toFixed(2)}</td><td>${cells.join("</td><td>")}</td></tr>`);
  }
  document.getElementById("arrows").innerHTML =
    "<tr><th></th><th>3D</th><th>Chen</th><th>Comp.</th><th>Scr.</th></tr>" + arrowRows.join("");
}

function showSection(event) {
  const point = event.points.find(pt => pt.data.name === "Sections");
  if (!point) return;
  const i = point.pointNumber;
  const el = document.getElementById("section");
  el.innerHTML = "";
  const head = document.createElement("b");
  head.textContent = data.sections[i];
  head.style.color = colorOf(i);
  el.append(head, document.createElement("br"), data.section_texts[i]);
}

const describe = () => document.getElementById("geometry").textContent = choice.value === "corrected"
  ? "Short sections keep more noise in their mean and would sit farther out for that reason alone; the length " +
    "of each mean is therefore estimated from pairs of distinct characters at least " + data.min_gap +
    " positions apart, which removes that noise."
  : "Plain section means: short sections keep more noise and sit farther out for that reason alone.";
choice.addEventListener("change", () => { describe(); render(); });
describe();
slider.addEventListener("input", render);
document.getElementById("path").addEventListener("change", render);  // select and checkboxes alike
document.getElementById("pairs").addEventListener("change", render);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
render();
document.getElementById("plot").on("plotly_hover", showSection);
document.getElementById("plot").on("plotly_click", showSection);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
