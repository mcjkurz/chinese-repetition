"""Echo curve: how strongly each character's stretch of text recurs elsewhere in the essay.

The whole essay is encoded in context once (overlapping chunks when it exceeds
the model's length limit), and the centred cosine matrix S of its --layer
character embeddings is computed. A repeated passage appears in S as a stripe
parallel to the main diagonal, offset by the distance between the two copies.
The echo of character i is the strength of the best such stripe through it:

1. similarity falls off steeply with distance, so the mean of each diagonal
   (all pairs d characters apart) is subtracted from it;
2. each diagonal is smoothed over --phrase characters, so that a match counts
   only when the characters around i also match the characters around j in
   the same order (aligned runs, not scattered identical characters);
3. echo(i) = max over j with |i - j| >= --min-lag of the smoothed value, so
   that i's own neighbourhood is excluded but a copy at any distance counts.

Unlike a sliding window, this does not depend on how far apart the copies of
a passage are. For each character the section holding its best match is
recorded too. The panels follow baguwen_window_similarity.py: (a) the original
essay, then each control set with its essays faintly and their mean in bold.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from qhchina import load_fonts
from scipy.ndimage import uniform_filter1d
from transformers import AutoModel, AutoTokenizer

from baguwen_window_similarity import align_to_reference, plot
from common import select_device
from make_baguwen_figure import SPAN_LABELS, label_bounds, section_bounds
from self_similarity_matrix import DEFAULT_MODEL, all_layer_character_embeddings, cosine_similarity_matrix

YLABEL = "Echo: best aligned match\nelsewhere in the essay"


def echo(matrix: np.ndarray, phrase: int, min_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-character echo strength and the position of its best match."""
    n = len(matrix)
    smoothed = np.full((n, n), -np.inf)
    for lag in range(min_lag, n):
        diagonal = np.diagonal(matrix, lag)
        diagonal = uniform_filter1d(diagonal - diagonal.mean(), phrase, mode="nearest")
        rows = np.arange(n - lag)
        smoothed[rows, rows + lag] = diagonal
        smoothed[rows + lag, rows] = diagonal
    return smoothed.max(axis=1), smoothed.argmax(axis=1)


def essay_echo(raw_text: str, args, model, tokenizer, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Positions, smoothed echo curve, best-match positions and section bounds of one essay."""
    text, bounds = section_bounds(raw_text)
    chars, hidden = all_layer_character_embeddings(text, model, tokenizer, device)
    if len(chars) != len(text):
        raise ValueError("Tokenization does not align with characters")
    strength, partner = echo(cosine_similarity_matrix(hidden[args.layer], center=True), args.phrase, args.min_lag)
    if args.smooth > 1:
        strength = uniform_filter1d(strength, args.smooth, mode="nearest")
    return np.arange(len(text), dtype=float), strength, partner, bounds


def partner_sections(partner: np.ndarray, bounds: list[int]) -> dict[str, dict[str, int]]:
    """For each labelled section, how many of its characters have their best match in each section."""
    spans = label_bounds(bounds)
    section = lambda position: SPAN_LABELS[int(np.searchsorted(spans, position, side="right")) - 1]
    return {label: dict(Counter(section(int(partner[i])) for i in range(start, end)).most_common())
            for label, start, end in zip(SPAN_LABELS, spans[:-1], spans[1:])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="data/4_eight_legged_essays/baguwen_chen_zilong.txt")
    parser.add_argument(
        "--controls",
        nargs="*",
        default=["(b) Composed parallel essays=data/4_eight_legged_essays/baguwen_controls/sensical.json",
                 "(c) Repetitive nonsense=data/4_eight_legged_essays/baguwen_controls/nonsense.json",
                 "(d) Complete scramble=data/4_eight_legged_essays/baguwen_controls/scrambled.json"],
        help="label=path pairs; each JSON has a 'passages' list of {id, text}",
    )
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--phrase", type=int, default=7, help="Length of the diagonal smoothing, in characters")
    parser.add_argument("--min-lag", type=int, default=10, help="Matches closer than this are ignored")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Width of the moving average applied to each curve, in positions (1 = none)")
    parser.add_argument("--shared-y", action="store_true", help="Give all panels the same y-axis range")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-dir", default="results/baguwen_echo_curve")
    args = parser.parse_args()

    load_fonts()
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    grid, original, partner, bounds = essay_echo(Path(args.original).read_text(encoding="utf-8"), args,
                                                 model, tokenizer, device)
    groups, control_partners = [], {}
    for spec in args.controls:
        label, path = spec.rsplit("=", 1)
        passages = json.loads(Path(path).read_text(encoding="utf-8"))["passages"]
        curves, partners = [], Counter()
        for passage in passages:
            x, y, passage_partner, passage_bounds = essay_echo(passage["text"], args, model, tokenizer, device)
            curves.append(align_to_reference(x, y, passage_bounds, bounds, grid))
            for section, counts in partner_sections(passage_partner, passage_bounds).items():
                partners.update({(section, target): count for target, count in counts.items()})
        groups.append((label, np.array(curves)))
        control_partners[label] = {section: {target: count for (s, target), count in partners.most_common() if s == section}
                                   for section in SPAN_LABELS}
        print(f"{label}: {len(curves)} essays")

    output_dir = Path(args.output_dir)
    stem = f"phrase{args.phrase}_minlag{args.min_lag}_layer{args.layer:02d}"
    plot(grid, original, groups, bounds, args, output_dir / f"{stem}.png", YLABEL,
         legend_loc="upper left")
    (output_dir / f"{stem}.json").write_text(json.dumps({
        "model": args.model, "original": args.original, "controls": args.controls, "layer": args.layer,
        "phrase": args.phrase, "min_lag": args.min_lag, "smooth": args.smooth, "section_bounds": bounds,
        "positions": grid.tolist(), "original_echo": original.tolist(),
        "original_partner_sections": partner_sections(partner, bounds),
        "control_mean_echo": {label: np.nanmean(curves, axis=0).tolist() for label, curves in groups},
        "control_partner_sections": control_partners,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
