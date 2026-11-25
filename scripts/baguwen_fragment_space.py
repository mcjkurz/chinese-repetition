"""The legs of an eight-legged essay as paths of punctuation-delimited fragments.

Each leg is split at Chinese punctuation; every fragment is encoded on its own
by GujiBERT and represented by the mean of its characters' embeddings (no
centring or other correction). Fragments are therefore of similar length
whatever the length of their leg. At every layer all fragments are projected
onto their first three principal components; each leg is drawn as a path of
its fragments in reading order, in its own colour, from light to dark.

Partner legs (1-2, 3-4, 5-6, 7-8) have the same number of fragments, so their
fragments correspond one to one, like the characters of a couplet. Transition
alignment is computed as in transition_analysis.py, at two levels:

- fragments: within each leg pair, the cosine between corresponding
  fragment-to-fragment transitions of the two legs, averaged over the pair;
- legs: each leg is the mean of its fragment vectors; the cosine between
  corresponding transitions of the first members (legs 1-3-5-7) and the
  second members (legs 2-4-6-8).

The baseline is position-matched random pairs: the same transition compared
with the corresponding transition of the partner leg in a different essay (the
composed essays, whose legs have the same numbers of fragments). Writes static
3D figures, a JSON file, and an interactive HTML page: a layer slider, legs that
can be shown, hidden or dimmed, opacity, size and label controls, and a
projection that can be refitted to the visible legs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_hex, to_rgb
from qhchina import load_fonts
from transformers import AutoModel, AutoTokenizer

from common import select_device
from self_similarity_matrix import DEFAULT_MODEL

LEG_PATTERN = re.compile(r"<leg>(.*?)</leg>")
FRAGMENT_SPLIT = re.compile(r"[，。；：？！、]")
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
LEG_COLORS = ["#1f4e8c", "#4aa3df", "#a6461a", "#e08a2c", "#1f7a5a", "#6cc08b", "#5b2c83", "#c45ab3"]
LINK_COLOR = "#8a8a92"


def leg_fragments(raw_text: str) -> list[list[str]]:
    return [[f for f in FRAGMENT_SPLIT.split(leg) if f] for leg in LEG_PATTERN.findall(raw_text)]


def fragment_vectors(fragments: list[str], model, tokenizer, device) -> np.ndarray:
    """Mean character embedding of each fragment, encoded on its own, at every layer: (layers, fragments, dim)."""
    encoded = tokenizer(fragments, return_tensors="pt", padding=True).to(device)
    lengths = encoded["attention_mask"].sum(dim=1).tolist()
    if [length - 2 for length in lengths] != [len(f) for f in fragments]:
        raise ValueError("Tokenization does not align with characters")
    with torch.no_grad():
        hidden = torch.stack(model(**encoded, output_hidden_states=True).hidden_states).float().cpu().numpy()
    return np.stack([hidden[:, i, 1:1 + len(f)].mean(axis=1) for i, f in enumerate(fragments)], axis=1)


def essay_vectors(raw_text: str, model, tokenizer, device) -> tuple[list[list[str]], list[np.ndarray]]:
    """Fragments of each leg and their vectors, one array (layers, fragments, dim) per leg."""
    legs = leg_fragments(raw_text)
    if len(legs) != 8:
        raise ValueError(f"Expected 8 legs, found {len(legs)}")
    flat = [f for leg in legs for f in leg]
    vectors = fragment_vectors(flat, model, tokenizer, device)
    bounds = np.cumsum([0] + [len(leg) for leg in legs])
    return legs, [vectors[:, start:end] for start, end in zip(bounds[:-1], bounds[1:])]


def unit_steps(path: np.ndarray) -> np.ndarray:
    """Unit transition vectors between consecutive points of a path (layers, points, dim)."""
    steps = np.diff(path, axis=1)
    return steps / np.linalg.norm(steps, axis=-1, keepdims=True)


def fragment_alignment(legs: list[np.ndarray]) -> np.ndarray:
    """Per layer and leg pair, the mean cosine of corresponding fragment transitions: (layers, pairs)."""
    return np.stack([np.sum(unit_steps(legs[a]) * unit_steps(legs[b]), axis=-1).mean(axis=-1)
                     for a, b in PAIRS], axis=1)


def leg_strings(legs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Unit transitions of the first-member and second-member strings of leg means: (layers, 3, dim) each."""
    means = np.stack([leg.mean(axis=1) for leg in legs], axis=1)
    return unit_steps(means[:, [a for a, _ in PAIRS]]), unit_steps(means[:, [b for _, b in PAIRS]])


def shades(color: str, count: int) -> list[str]:
    """Shades of a colour from light (first) to full (last)."""
    rgb = np.array(to_rgb(color))
    return [to_hex(rgb + (1 - rgb) * mix) for mix in np.linspace(0.6, 0.0, count)]


def project(legs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D PCA coordinates of all fragments and of the leg means, per layer, with variance shares."""
    fragments = np.concatenate(legs, axis=1)
    means = np.stack([leg.mean(axis=1) for leg in legs], axis=1)
    coordinates, mean_coordinates, shares, previous = [], [], [], None
    for layer in range(fragments.shape[0]):
        centre = fragments[layer].mean(axis=0)
        _, s, vt = np.linalg.svd(fragments[layer] - centre, full_matrices=False)
        axes = vt[:3]
        points = (fragments[layer] - centre) @ axes.T
        if previous is not None:
            signs = np.sign(np.sum(points * previous, axis=0))
            signs[signs == 0] = 1
            points, axes = points * signs, axes * signs[:, None]
        coordinates.append(points)
        mean_coordinates.append((means[layer] - centre) @ axes.T)
        shares.append((s ** 2 / np.sum(s ** 2))[:3])
        previous = points
    return np.array(coordinates), np.array(mean_coordinates), np.array(shares)


def plot_static(points: np.ndarray, legs: list[list[str]], shares: np.ndarray, layer: int, output: Path,
                dpi: int) -> None:
    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect(None, zoom=0.85)
    bounds = np.cumsum([0] + [len(leg) for leg in legs])
    for a, b in PAIRS:
        for i in range(len(legs[a])):
            ax.plot(*np.stack([points[bounds[a] + i], points[bounds[b] + i]]).T, color=LINK_COLOR,
                    linewidth=0.5, alpha=0.6)
    for index, (leg, color) in enumerate(zip(legs, LEG_COLORS)):
        path = points[bounds[index]:bounds[index + 1]]
        ax.plot(*path.T, color=color, linewidth=1.6, label=f"Leg {index + 1}")
        ax.scatter(*path.T, color=shades(color, len(leg)), s=28, depthshade=False, edgecolors=color, linewidths=0.5)
    for axis, share, setter in zip("123", shares, (ax.set_xlabel, ax.set_ylabel, ax.set_zlabel)):
        setter(f"PC{axis} ({share:.0%})")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.set_title(f"Chen Zilong: leg fragments, GujiBERT layer {layer}\n"
                 "light to dark = reading order; grey = corresponding fragments of partner legs", fontsize=10)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="data/4_eight_legged_essays/baguwen_chen_zilong.txt")
    parser.add_argument("--baseline", default="data/4_eight_legged_essays/baguwen_controls/sensical.json",
                        help="Essays whose partner legs give the position-matched random-pair baseline")
    parser.add_argument("--figure-layers", type=int, nargs="+", default=[8, 12],
                        help="Layers drawn as static figures; the first is the page's starting layer")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-dir", default="results/baguwen_fragment_space")
    args = parser.parse_args()

    load_fonts()
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    texts, legs = essay_vectors(Path(args.original).read_text(encoding="utf-8"), model, tokenizer, device)
    others = []
    for passage in json.loads(Path(args.baseline).read_text(encoding="utf-8"))["passages"]:
        other_texts, other_legs = essay_vectors(passage["text"], model, tokenizer, device)
        if [len(leg) for leg in other_texts] == [len(leg) for leg in texts]:
            others.append(other_legs)

    # Fragments: Chen's first leg of a pair against the second leg of the same pair in another essay.
    fragment_observed = fragment_alignment(legs)
    fragment_random = np.array([np.stack([
        np.sum(unit_steps(legs[a]) * unit_steps(other[b]), axis=-1).mean(axis=-1) for a, b in PAIRS], axis=1)
        for other in others])
    fragment_baseline = fragment_random.mean(axis=0)
    odd, even = leg_strings(legs)
    leg_observed = np.sum(odd * even, axis=-1).mean(axis=-1)
    leg_random = np.array([np.sum(odd * leg_strings(other)[1], axis=-1).mean(axis=-1) for other in others])
    leg_baseline = leg_random.mean(axis=0)
    # Share of baseline essays whose random pairing reaches Chen's alignment (an empirical p-value).
    fragment_p = (np.sum(fragment_random >= fragment_observed, axis=0) + 1) / (len(others) + 1)
    leg_p = (np.sum(leg_random >= leg_observed, axis=0) + 1) / (len(others) + 1)

    pair_names = [f"Legs {a + 1}-{b + 1}" for a, b in PAIRS]
    print(f"Baseline essays: {len(others)}")
    print("layer  " + "  ".join(f"{name:>16}" for name in pair_names) + f"  {'Strings of legs':>16}")
    print("       " + "  ".join(f"{'aligned / base':>16}" for _ in range(len(PAIRS) + 1)))
    for layer in range(len(leg_observed)):
        cells = [f"{fragment_observed[layer, k]:6.2f}{fragment_baseline[layer, k]:6.2f} p{fragment_p[layer, k]:.2f}"
                 for k in range(len(PAIRS))]
        print(f"{layer:5d}  " + "  ".join(cells) + f"  {leg_observed[layer]:6.2f}{leg_baseline[layer]:6.2f} p{leg_p[layer]:.2f}")

    coordinates, mean_coordinates, shares = project(legs)
    output_dir = Path(args.output_dir)
    for layer in args.figure_layers:
        plot_static(coordinates[layer], texts, shares[layer], layer,
                    output_dir / f"fragment_space_layer{layer:02d}.png", args.dpi)

    payload = {
        "model": args.model, "original": args.original, "baseline": args.baseline, "baseline_essays": len(others),
        "fragments": texts, "pairs": PAIRS, "pair_names": pair_names, "leg_colors": LEG_COLORS,
        "fragment_colors": [shades(color, len(leg)) for color, leg in zip(LEG_COLORS, texts)],
        "link_color": LINK_COLOR, "default_layer": args.figure_layers[0],
        "coordinates": coordinates.round(4).tolist(), "leg_mean_coordinates": mean_coordinates.round(4).tolist(),
        "variance_shares": shares.round(4).tolist(),
        # Full fragment vectors, so that the page can refit the projection to the legs shown.
        "vectors": np.concatenate(legs, axis=1).round(3).tolist(),
        "fragment_alignment": fragment_observed.round(4).tolist(),
        "fragment_baseline": fragment_baseline.round(4).tolist(),
        "fragment_baseline_sd": fragment_random.std(axis=0).round(4).tolist(),
        "fragment_p": fragment_p.round(4).tolist(),
        "leg_alignment": leg_observed.round(4).tolist(), "leg_baseline": leg_baseline.round(4).tolist(),
        "leg_baseline_sd": leg_random.std(axis=0).round(4).tolist(), "leg_p": leg_p.round(4).tolist(),
    }
    # The fragment vectors are only needed by the page.
    (output_dir / "fragment_space.json").write_text(
        json.dumps({key: value for key, value in payload.items() if key != "vectors"}, ensure_ascii=False, indent=2)
        + "\n", encoding="utf-8")
    (output_dir / "fragment_space.html").write_text(
        HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
    print(f"Saved: {output_dir / 'fragment_space.html'}")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Leg fragments</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
  :root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6b6b6b; --line: #e3e3e6; --panel: #f6f6f8; --accent: #1f4e8c; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) { --bg: #16171a; --fg: #ececf0; --muted: #a0a0a8; --line: #2c2d33; --panel: #1f2025; --accent: #8fb8ea; }
  }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 15px/1.5 -apple-system, "Segoe UI", "PingFang TC", "Noto Sans TC", sans-serif; }
  main { max-width: 1320px; margin: 0 auto; padding: 20px 16px 40px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  p.lede { color: var(--muted); margin: 0 0 16px; max-width: 85ch; }
  .layout { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 20px; align-items: start; }
  @media (max-width: 920px) { .layout { grid-template-columns: 1fr; } }
  #plot { height: 760px; border: 1px solid var(--line); border-radius: 8px; }
  aside { display: flex; flex-direction: column; gap: 12px; }
  .card { background: var(--panel); border-radius: 8px; padding: 12px 14px; }
  .card h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 0 0 8px; }
  label { display: block; margin: 3px 0; }
  .row { display: grid; grid-template-columns: 110px 1fr 38px; gap: 8px; align-items: center; margin: 4px 0; }
  .row span:last-child { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); font-size: 13px; }
  input[type=range] { width: 100%; }
  .legs { display: grid; grid-template-columns: 1fr 1fr; gap: 2px 12px; }
  .legs label { display: flex; align-items: center; gap: 6px; }
  .swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
  .buttons { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 4px; }
  button { font: inherit; font-size: 13px; padding: 3px 9px; border-radius: 6px; border: 1px solid var(--line);
           background: var(--bg); color: var(--fg); cursor: pointer; }
  button:hover { border-color: var(--accent); }
  select { font: inherit; font-size: 14px; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
  th, td { text-align: right; padding: 3px 4px; border-bottom: 1px solid var(--line); }
  th:first-child, td:first-child { text-align: left; }
  #fragment { font-size: 14px; min-height: 3em; }
  .note { color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<main>
  <h1>Chen Zilong's legs as paths of fragments</h1>
  <p class="lede">Each leg is split at punctuation; every fragment is encoded on its own by GujiBERT and shown as the
  mean of its characters, projected onto three principal components. Each leg has its own colour, partner legs
  related colours; fragments run from light to dark in reading order. Grey links join corresponding fragments of
  partner legs. Drag to rotate; hover or click a point to read its fragment.</p>
  <div class="layout">
    <div id="plot"></div>
    <aside>
      <div class="card">
        <h2>Layer <span id="layer-value"></span></h2>
        <input id="layer" type="range" min="0" step="1">
        <label>Projection <select id="projection">
          <option value="all">Fitted to all fragments</option>
          <option value="visible">Refitted to the visible legs</option>
        </select></label>
        <div class="note" id="variance"></div>
        <div class="buttons"><button id="reset">Reset view</button></div>
      </div>
      <div class="card">
        <h2>Legs</h2>
        <div class="legs" id="legs"></div>
        <div class="buttons" id="presets"></div>
        <label>Other legs <select id="others">
          <option value="hide">Hidden</option>
          <option value="dim">Dimmed</option>
        </select></label>
        <div class="row"><span>Dimmed opacity</span><input id="dim" type="range" min="0.02" max="0.6" step="0.02" value="0.12"><span></span></div>
      </div>
      <div class="card">
        <h2>Display</h2>
        <label><input id="fragments" type="checkbox" checked> Fragments</label>
        <label><input id="paths" type="checkbox" checked> Paths within each leg</label>
        <label><input id="links" type="checkbox" checked> Links between corresponding fragments</label>
        <label>Leg means <select id="means">
          <option value="none">Off</option>
          <option value="pairs">Partner legs joined (1–2, 3–4, 5–6, 7–8)</option>
          <option value="strings">Odd and even strings (1-3-5-7, 2-4-6-8)</option>
          <option value="both">Pairs and strings</option>
          <option value="order">Legs in order (1 → 8)</option>
        </select></label>
        <div class="row"><span>Leg opacity</span><input id="alpha" type="range" min="0.1" max="1" step="0.05" value="1"><span></span></div>
        <div class="row"><span>Link opacity</span><input id="link-alpha" type="range" min="0.05" max="1" step="0.05" value="0.6"><span></span></div>
        <div class="row"><span>Point size</span><input id="size" type="range" min="2" max="14" step="1" value="6"><span></span></div>
        <div class="row"><span>Line width</span><input id="width" type="range" min="1" max="12" step="1" value="5"><span></span></div>
        <label>Labels <select id="labels">
          <option value="none">None</option>
          <option value="number">Fragment number</option>
          <option value="text">Fragment text</option>
        </select></label>
      </div>
      <div class="card">
        <h2>Transition alignment</h2>
        <table id="alignment"></table>
        <p class="note">Mean cosine between corresponding transitions in the full space (unaffected by the
        projection). Fragment rows: consecutive fragments within the two partner legs. Last row: consecutive leg
        means of legs 1-3-5-7 against 2-4-6-8. Baseline: the same transitions against the partner leg of the same
        slot in other essays (composed, <span id="n"></span> essays); p: share of those essays whose pairing reaches
        Chen's own (smallest possible value about 0.02).</p>
      </div>
      <div class="card">
        <h2>Fragment</h2>
        <div id="fragment">Hover over a point.</div>
      </div>
    </aside>
  </div>
</main>
<script>
const data = __DATA__;
const $ = id => document.getElementById(id);
const slider = $("layer");
slider.max = data.coordinates.length - 1;
slider.value = data.default_layer;
$("n").textContent = data.baseline_essays;
const legCount = data.fragments.length;
const starts = data.fragments.reduce((acc, leg) => [...acc, acc[acc.length - 1] + leg.length], [0]);
const dark = () => matchMedia("(prefers-color-scheme: dark)").matches;
const checked = id => $(id).checked;

// Leg checkboxes and presets.
const selected = new Set(Array.from({length: legCount}, (_, k) => k));
data.fragments.forEach((_, k) => {
  const label = document.createElement("label");
  label.innerHTML = `<input type="checkbox" data-leg="${k}" checked><span class="swatch" style="background:${data.leg_colors[k]}"></span>Leg ${k + 1}`;
  $("legs").append(label);
});
$("legs").addEventListener("change", e => {
  const k = +e.target.dataset.leg;
  e.target.checked ? selected.add(k) : selected.delete(k);
  render();
});
const presets = [["All", [0, 1, 2, 3, 4, 5, 6, 7]], ["None", []], ...data.pairs.map(([a, b]) => [`${a + 1}–${b + 1}`, [a, b]]),
                 ["Odd", data.pairs.map(([a]) => a)], ["Even", data.pairs.map(([, b]) => b)]];
presets.forEach(([name, legs]) => {
  const button = document.createElement("button");
  button.textContent = name;
  button.addEventListener("click", () => {
    selected.clear(); legs.forEach(k => selected.add(k));
    document.querySelectorAll("#legs input").forEach(box => { box.checked = selected.has(+box.dataset.leg); });
    render();
  });
  $("presets").append(button);
});

// PCA of the chosen fragments: eigenvectors of the small Gram matrix (Jacobi rotations).
function eigenSymmetric(matrix) {
  const n = matrix.length, a = matrix.map(row => row.slice()), v = a.map((_, i) => a.map((_, j) => +(i === j)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-12) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-15) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = v[k][p], vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  return {values: a.map((row, i) => row[i]), vectors: v};
}

const cache = new Map();
function projection(layer) {
  if ($("projection").value === "all" || selected.size === 0) {
    return {points: data.coordinates[layer], means: data.leg_mean_coordinates[layer], shares: data.variance_shares[layer]};
  }
  const key = `${layer}:${[...selected].sort().join(",")}`;
  if (cache.has(key)) return cache.get(key);
  const vectors = data.vectors[layer];
  const rows = [...selected].sort().flatMap(k => Array.from({length: data.fragments[k].length}, (_, i) => starts[k] + i));
  const dim = vectors[0].length;
  const centre = new Array(dim).fill(0);
  rows.forEach(r => vectors[r].forEach((x, j) => { centre[j] += x / rows.length; }));
  const centred = rows.map(r => vectors[r].map((x, j) => x - centre[j]));
  const gram = centred.map(a => centred.map(b => a.reduce((sum, x, j) => sum + x * b[j], 0)));
  const {values, vectors: eig} = eigenSymmetric(gram);
  const order = values.map((_, i) => i).sort((i, j) => values[j] - values[i]).slice(0, 3);
  const total = values.reduce((sum, x) => sum + Math.max(x, 0), 0);
  // Principal axes in the embedding space, so that every fragment and leg mean can be projected.
  const axes = order.map(i => {
    const axis = new Array(dim).fill(0);
    centred.forEach((row, r) => row.forEach((x, j) => { axis[j] += x * eig[r][i]; }));
    const norm = Math.hypot(...axis) || 1;
    return axis.map(x => x / norm);
  });
  const reference = data.coordinates[layer];
  const project = vec => axes.map(axis => vec.reduce((sum, x, j) => sum + (x - centre[j]) * axis[j], 0));
  let points = vectors.map(project);
  // Match each axis's sign to the all-fragment projection, so that switching keeps the orientation.
  const signs = [0, 1, 2].map(c => Math.sign(rows.reduce((sum, r) => sum + points[r][c] * reference[r][c], 0)) || 1);
  points = points.map(p => p.map((x, c) => x * signs[c]));
  const meanVectors = data.fragments.map((_, k) => {
    const mean = new Array(dim).fill(0);
    for (let i = starts[k]; i < starts[k + 1]; i++) vectors[i].forEach((x, j) => { mean[j] += x / (starts[k + 1] - starts[k]); });
    return mean;
  });
  const result = {points, means: meanVectors.map(project).map(p => p.map((x, c) => x * signs[c])),
                  shares: order.map(i => values[i] / total)};
  cache.set(key, result);
  return result;
}

function traces(layer) {
  const {points: p, means: m} = projection(layer);
  const dim = $("others").value === "dim";
  const visible = k => selected.has(k) || dim;
  const alphaOf = k => selected.has(k) ? +$("alpha").value : +$("dim").value;
  const size = +$("size").value, width = +$("width").value, labels = $("labels").value;
  const out = [];
  if (checked("fragments") && checked("links")) {
    const x = [], y = [], z = [];
    data.pairs.filter(([a, b]) => selected.has(a) && selected.has(b)).forEach(([a, b]) =>
      data.fragments[a].forEach((_, i) => {
        const u = p[starts[a] + i], v = p[starts[b] + i];
        x.push(u[0], v[0], null); y.push(u[1], v[1], null); z.push(u[2], v[2], null);
      }));
    out.push({type: "scatter3d", mode: "lines", name: "Corresponding fragments", x, y, z,
              line: {color: data.link_color, width: Math.max(1, width / 2.5)}, opacity: +$("link-alpha").value,
              hoverinfo: "skip"});
  }
  data.fragments.forEach((leg, k) => {
    if (!checked("fragments") || !visible(k)) return;
    const pts = p.slice(starts[k], starts[k + 1]);
    const mode = (checked("paths") ? "lines+markers" : "markers") + (labels === "none" ? "" : "+text");
    out.push({type: "scatter3d", mode, name: `Leg ${k + 1}`, opacity: alphaOf(k), showlegend: selected.has(k),
              x: pts.map(q => q[0]), y: pts.map(q => q[1]), z: pts.map(q => q[2]),
              line: {color: data.leg_colors[k], width},
              marker: {size, color: data.fragment_colors[k], line: {color: data.leg_colors[k], width: 1}},
              text: leg.map((f, i) => labels === "text" ? f : labels === "number" ? `${k + 1}.${i + 1}` : ""),
              textfont: {color: data.leg_colors[k], size: 12}, textposition: "top center",
              customdata: leg.map((f, i) => [k, i, `Leg ${k + 1}, fragment ${i + 1}: ${f}`]),
              hovertemplate: "%{customdata[2]}<extra></extra>"});
  });
  const means = $("means").value;
  if (means !== "none") {
    const line = (name, members, color, lineWidth) => {
      const shown = members.filter(k => selected.has(k));
      if (shown.length < 2) return;
      out.push({type: "scatter3d", mode: "lines", name, x: shown.map(i => m[i][0]), y: shown.map(i => m[i][1]),
                z: shown.map(i => m[i][2]), line: {color, width: lineWidth}, opacity: +$("alpha").value,
                hoverinfo: "skip"});
    };
    if (means === "pairs" || means === "both") {
      data.pairs.forEach(([a, b]) => line(`Mean of legs ${a + 1}–${b + 1}`, [a, b], data.leg_colors[a], width + 5));
    }
    if (means === "strings" || means === "both") {
      line("Legs 1-3-5-7", data.pairs.map(([a]) => a), "#2f2f35", width + 1);
      line("Legs 2-4-6-8", data.pairs.map(([, b]) => b), "#9a9aa2", width + 1);
    }
    if (means === "order") line("Legs 1 → 8", data.fragments.map((_, k) => k), "#6b6b73", width + 1);
    const shown = data.fragments.map((_, k) => k).filter(k => selected.has(k));
    out.push({type: "scatter3d", mode: labels === "none" ? "markers" : "markers+text", name: "Leg means",
              x: shown.map(k => m[k][0]), y: shown.map(k => m[k][1]), z: shown.map(k => m[k][2]),
              marker: {size: size + 4, symbol: "diamond", color: shown.map(k => data.leg_colors[k]),
                       line: {color: dark() ? "#ececf0" : "#1d1d1f", width: 1}},
              text: shown.map(k => `Leg ${k + 1}`), textposition: "top center",
              textfont: {color: shown.map(k => data.leg_colors[k]), size: 13},
              customdata: shown.map(k => `Mean of the ${data.fragments[k].length} fragments of leg ${k + 1}`),
              hovertemplate: "%{customdata}<extra></extra>"});
  }
  return out;
}

let revision = 0;
function layout(shares) {
  const fg = dark() ? "#ececf0" : "#1d1d1f", grid = dark() ? "#34353b" : "#e3e3e6";
  const axis = t => ({title: {text: t}, color: fg, gridcolor: grid, zerolinecolor: grid, showbackground: false});
  return {uirevision: `keep${revision}`, paper_bgcolor: "rgba(0,0,0,0)", font: {color: fg},
          margin: {l: 0, r: 0, t: 0, b: 0}, legend: {x: 0, y: 1, bgcolor: "rgba(0,0,0,0)"},
          scene: {xaxis: axis(`PC1 (${Math.round(shares[0] * 100)}%)`), yaxis: axis(`PC2 (${Math.round(shares[1] * 100)}%)`),
                  zaxis: axis(`PC3 (${Math.round(shares[2] * 100)}%)`), aspectmode: "cube"}};
}

function render() {
  const layer = +slider.value;
  $("layer-value").textContent = layer;
  document.querySelectorAll(".row").forEach(row => { row.lastElementChild.textContent = row.querySelector("input").value; });
  const {shares} = projection(layer);
  const fitted = $("projection").value === "visible" && selected.size > 0 ? "the visible legs' fragments" : "all fragments";
  $("variance").textContent =
    `The three components hold ${Math.round((shares[0] + shares[1] + shares[2]) * 100)}% of the variance of ${fitted}.`;
  Plotly.react("plot", traces(layer), layout(shares), {responsive: true, displaylogo: false});
  const row = (name, observed, base, p) =>
    `<tr><td>${name}</td><td>${observed.toFixed(2)}</td><td>${base.toFixed(2)}</td><td>${(observed - base).toFixed(2)}</td>` +
    `<td>${p.toFixed(2)}</td></tr>`;
  const rows = data.pair_names.map((name, k) =>
    row(`Fragments, ${name.toLowerCase()}`, data.fragment_alignment[layer][k], data.fragment_baseline[layer][k],
        data.fragment_p[layer][k]));
  rows.push(row("Leg means, 1-3-5-7 / 2-4-6-8", data.leg_alignment[layer], data.leg_baseline[layer], data.leg_p[layer]));
  $("alignment").innerHTML = "<tr><th></th><th>Aligned</th><th>Base</th><th>Diff.</th><th>p</th></tr>" + rows.join("");
}

function showFragment(event) {
  const point = event.points.find(pt => pt.customdata && pt.customdata.length === 3);
  if (!point) return;
  const [k, i] = point.customdata;
  const el = $("fragment");
  el.innerHTML = "";
  const head = document.createElement("b");
  head.textContent = `Leg ${k + 1}, fragment ${i + 1}`;
  head.style.color = data.leg_colors[k];
  const partner = data.pairs.find(pair => pair.includes(k));
  const other = partner[0] === k ? partner[1] : partner[0];
  const note = document.createElement("div");
  note.className = "note";
  note.textContent = `Corresponds to leg ${other + 1}: ${data.fragments[other][i]}`;
  el.append(head, document.createElement("br"), data.fragments[k][i], note);
}

slider.addEventListener("input", render);
["fragments", "paths", "links", "means", "others", "labels", "projection"].forEach(id => $(id).addEventListener("change", render));
["alpha", "link-alpha", "size", "width", "dim"].forEach(id => $(id).addEventListener("input", render));
$("reset").addEventListener("click", () => { revision += 1; render(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
render();
$("plot").on("plotly_hover", showFragment);
$("plot").on("plotly_click", showFragment);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
