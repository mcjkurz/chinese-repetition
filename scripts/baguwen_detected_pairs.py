"""Formally parallel passages in 欽定四書文 against the same form in non-parallel prose.

All texts are punctuated by the same model (prepare_qinding_sishuwen.py), so
that fragments are cut the same way in essays and prose. Parallel pairs are
then found by form alone, as the couplet study defines parallelism by form:
two consecutive passages A and B of n fragments each (n from --min-n to
--max-n) whose corresponding fragments have nearly the same length (mean
absolute difference at most --mean-gap characters, none more than --max-gap).
Scanning each text from the start, the longest such pair at a position is
taken and the scan resumes after it. The rule looks neither at the characters
nor at the embeddings.

The same rule is applied to 古文觀止 (volumes 8-12) and to a sample of 明史
biographies of similar size, after their own punctuation is removed and
replaced by the model's. Detected pairs in prose are passages that happen to
share the form without being written as parallel; they play the role of the
outer couplets. The rule is checked against the leg pairs marked by hand in six
essays (Chen Zilong's and data/4_eight_legged_essays/baguwen_corpus.json).

Every fragment is encoded on its own by GujiBERT (mean of its characters), and
the alignment of a pair is the mean cosine between corresponding
fragment-to-fragment transitions, as in baguwen_pseudo_legs.py. The baseline
is position-matched random pairs: leg A against leg B of --random-pairs other
detected pairs with the same n, drawn from all corpora together. The share of
characters that corresponding fragments have in common is recorded for every
pair, so that parallelism carried by shared frame words can be told apart from
parallelism of different words; alignment is reported by bins of that share,
and essays are ranked by their number of pairs built on a shared frame
(--frame-share), as candidates for marking legs by hand.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qhchina import load_fonts
from scipy.stats import mannwhitneyu
from transformers import AutoModel, AutoTokenizer

from baguwen_pseudo_legs import alignment, cohens_d, pair_steps
from common import select_device
from prepare_qinding_sishuwen import BREAKS, HAN, Punctuator
from self_similarity_matrix import DEFAULT_MODEL, parse_leg_marked_text

COLORS = {"欽定四書文": "#08306b", "古文觀止": "#555559", "明史": "#a4a4ab"}


def fragments_with_spans(punctuated: str) -> list[tuple[str, int, int]]:
    """Fragments of a punctuated text with their start and end in the unpunctuated characters."""
    out, current, position = [], "", 0
    for char in punctuated:
        if char in BREAKS:
            if current:
                out.append((current, position - len(current), position))
            current = ""
        elif HAN.match(char):
            current += char
            position += 1
    if current:
        out.append((current, position - len(current), position))
    return out


def detect_pairs(fragments: list[tuple[str, int, int]], args) -> list[tuple[list, list]]:
    """Consecutive passages of n fragments each whose corresponding fragments have nearly equal lengths."""
    pairs, start = [], 0
    while start + 2 * args.min_n <= len(fragments):
        found = None
        for n in range(min(args.max_n, (len(fragments) - start) // 2), args.min_n - 1, -1):
            a, b = fragments[start:start + n], fragments[start + n:start + 2 * n]
            gaps = [abs(len(x[0]) - len(y[0])) for x, y in zip(a, b)]
            if np.mean(gaps) <= args.mean_gap and max(gaps) <= args.max_gap and \
                    all(x[0] != y[0] for x, y in zip(a[:-1], a[1:])) and all(x[0] != y[0] for x, y in zip(b[:-1], b[1:])):
                found = (a, b)
                break
        if found:
            pairs.append(found)
            start += 2 * len(found[0])
        else:
            start += 1
    return pairs


def shared_characters(a: list[str], b: list[str]) -> float:
    """Mean share of characters that corresponding fragments have in common."""
    return float(np.mean([len(set(x) & set(y)) / max(len(set(x)), len(set(y))) for x, y in zip(a, b)]))


def plain(text: str) -> str:
    return "".join(HAN.findall(text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--essays", default="data/4_eight_legged_essays/qinding_sishuwen.json")
    parser.add_argument("--guwen", default="data/4_eight_legged_essays/pseudo_legs/guwen.json")
    parser.add_argument("--mingshi", default="data/4_eight_legged_essays/pseudo_legs/mingshi.json")
    parser.add_argument("--hand-marked", default="data/4_eight_legged_essays/baguwen_corpus.json")
    parser.add_argument("--chen", default="data/4_eight_legged_essays/baguwen_chen_zilong.txt")
    parser.add_argument("--min-n", type=int, default=3)
    parser.add_argument("--max-n", type=int, default=30)
    parser.add_argument("--mean-gap", type=float, default=1.0)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--random-pairs", type=int, default=30)
    parser.add_argument("--frame-share", type=float, default=0.25,
                        help="Pairs whose corresponding fragments share at least this share of characters count as frame pairs")
    parser.add_argument("--figure-layer", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pairs-per-batch", type=int, default=32)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-dir", default="results/baguwen_detected_pairs")
    args = parser.parse_args()

    load_fonts()
    device = select_device(args.device)
    rng = random.Random(args.seed)
    punctuate = Punctuator(device)

    # Texts, all punctuated by the model.
    essays = json.loads(Path(args.essays).read_text(encoding="utf-8"))["essays"]
    texts = {"欽定四書文": [(f"{e['period']}/{e['title']}/{e['author']}", e["punctuated"]) for e in essays]}
    target = sum(len(e["text"]) for e in essays)
    guwen = json.loads(Path(args.guwen).read_text(encoding="utf-8"))["pieces"]
    texts["古文觀止"] = [(p["title"], punctuate(plain(p["text"]))) for p in guwen]
    mingshi = json.loads(Path(args.mingshi).read_text(encoding="utf-8"))["pieces"]
    rng.shuffle(mingshi)
    sample, size = [], 0
    for piece in mingshi:
        if size >= target:
            break
        sample.append(piece)
        size += len(plain(piece["text"]))
    texts["明史"] = [(p["title"], punctuate(plain(p["text"]))) for p in sample]

    # The rule against the hand-marked pairs: is each marked pair found, both members overlapping a detected pair?
    hand = json.loads(Path(args.hand_marked).read_text(encoding="utf-8"))["essays"]
    chen_text, chen_legs, _ = parse_leg_marked_text(Path(args.chen).read_text(encoding="utf-8"))
    chen_pairs = [(chen_legs[i], chen_legs[i + 1]) for i in range(0, 8, 2)]
    marked = [(plain(e["text"]), [(plain(e["text"]).index(plain(p["a"])), len(plain(p["a"])),
                                   plain(e["text"]).index(plain(p["b"])), len(plain(p["b"]))) for p in e["pairs"]
                                  if p["fragments"] >= args.min_n]) for e in hand]
    chen_plain = plain(chen_text)
    marked.append((chen_plain, [(chen_plain.index(plain(chen_text[a0:a1])), a1 - a0,
                                 chen_plain.index(plain(chen_text[b0:b1])), b1 - b0) for (a0, a1), (b0, b1) in chen_pairs]))
    found = total = 0
    for text, spans in marked:
        detected = detect_pairs(fragments_with_spans(punctuate(text)), args)
        for a_start, a_len, b_start, b_len in spans:
            total += 1
            covers = lambda start, length, member: len(set(range(start, start + length)) &
                                                       set(range(member[0][1], member[-1][2]))) >= 0.5 * length
            found += any(covers(a_start, a_len, a) and covers(b_start, b_len, b) for a, b in detected)
    print(f"Hand-marked pairs found by the rule: {found} of {total}")

    # Detected pairs in every corpus.
    detected, rates = {}, {}
    for label, items in texts.items():
        pairs = []
        for name, punctuated in items:
            for a, b in detect_pairs(fragments_with_spans(punctuated), args):
                pairs.append({"text": name, "a": [f[0] for f in a], "b": [f[0] for f in b]})
        characters = sum(len(plain(p)) for _, p in items)
        covered = sum(len("".join(p["a"] + p["b"])) for p in pairs)
        rates[label] = {"texts": len(items), "characters": characters, "pairs": len(pairs),
                        "pairs_per_1000_characters": 1000 * len(pairs) / characters, "share_of_text_in_pairs": covered / characters,
                        "sizes": dict(sorted(Counter(len(p["a"]) for p in pairs).items()))}
        detected[label] = pairs
        print(f"{label}: {len(items)} texts, {characters} characters, {len(pairs)} pairs "
              f"({rates[label]['pairs_per_1000_characters']:.2f} per 1000 characters, "
              f"{100 * rates[label]['share_of_text_in_pairs']:.1f}% of the text)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    records = {}
    for label, pairs in detected.items():
        steps = pair_steps([(p["a"], p["b"]) for p in pairs], args, model, tokenizer, device)
        info = {tuple(p["a"]) + tuple(p["b"]): p for p in pairs}
        for record in steps:
            record["text"] = info[tuple(record["a"]) + tuple(record["b"])]["text"]
            record["shared"] = shared_characters(record["a"], record["b"])
        records[label] = steps
    pool: dict[int, list[dict]] = {}
    for steps in records.values():
        for record in steps:
            pool.setdefault(len(record["a"]), []).append(record)
    for steps in records.values():
        for record in steps:
            others = [o for o in pool[len(record["a"])] if o is not record]
            chosen = rng.sample(others, min(args.random_pairs, len(others)))
            record["alignment"] = alignment(record["A"], record["B"])
            record["baseline"] = np.mean([alignment(record["A"], o["B"]) for o in chosen], axis=0) if chosen else \
                np.zeros_like(record["alignment"])

    layers = len(next(iter(records.values()))[0]["alignment"])
    L = args.figure_layer
    diff = {label: np.array([r["alignment"] - r["baseline"] for r in steps]) for label, steps in records.items()}
    shared = {label: np.array([r["shared"] for r in steps]) for label, steps in records.items()}
    summary = {}
    essay_label = "欽定四書文"
    low = shared[essay_label] <= np.quantile(shared[essay_label], 0.25)
    for label, values in diff.items():
        entry = {"pairs": len(values), "mean_shared_characters": float(shared[label].mean()),
                 "difference": values.mean(axis=0).tolist(), "difference_sd": values.std(axis=0).tolist(),
                 "alignment": np.array([r["alignment"] for r in records[label]]).mean(axis=0).tolist(),
                 "baseline": np.array([r["baseline"] for r in records[label]]).mean(axis=0).tolist(), **rates[label]}
        if label != essay_label:
            entry["essays_vs_this_cohens_d"] = [cohens_d(diff[essay_label][:, k], values[:, k]) for k in range(layers)]
            entry["essays_vs_this_mann_whitney_p"] = [float(mannwhitneyu(diff[essay_label][:, k], values[:, k],
                                                                         alternative="greater").pvalue) for k in range(layers)]
            entry["low_overlap_essays_vs_this_cohens_d"] = [cohens_d(diff[essay_label][low, k], values[:, k]) for k in range(layers)]
        summary[label] = entry
    summary[essay_label]["low_overlap_quartile"] = {
        "max_shared_characters": float(shared[essay_label][low].max()), "pairs": int(low.sum()),
        "difference": diff[essay_label][low].mean(axis=0).tolist()}
    # Alignment at equal levels of shared characters, and how often each level occurs.
    bins = [0, 0.05, 0.15, 0.25, 0.4, 1.01]
    for label in records:
        summary[label]["by_shared_characters_at_figure_layer"] = [
            {"from": low_edge, "to": high_edge, "pairs": int(((shared[label] >= low_edge) & (shared[label] < high_edge)).sum()),
             "difference": float(diff[label][(shared[label] >= low_edge) & (shared[label] < high_edge), L].mean())
             if ((shared[label] >= low_edge) & (shared[label] < high_edge)).any() else None}
            for low_edge, high_edge in zip(bins[:-1], bins[1:])]
    # Essays with the most pairs built on a shared frame: candidates for marking legs by hand.
    frame_pairs = Counter(r["text"] for r in records[essay_label] if r["shared"] >= args.frame_share)
    candidates = [{"essay": name, "frame_pairs": count} for name, count in frame_pairs.most_common()]
    by_period = {}
    for period in ["化治", "正嘉", "隆萬", "啟禎", "本朝"]:
        values = np.array([r["alignment"][L] - r["baseline"][L] for r in records[essay_label] if r["text"].startswith(period)])
        by_period[period] = {"pairs": len(values), "difference": float(values.mean()) if len(values) else None}
    summary[essay_label]["by_period_at_figure_layer"] = by_period

    for label, entry in summary.items():
        extra = ""
        if "essays_vs_this_cohens_d" in entry:
            extra = (f"; essays vs this: d {entry['essays_vs_this_cohens_d'][L]:.2f} (p={entry['essays_vs_this_mann_whitney_p'][L]:.1e}),"
                     f" low-overlap essays vs this: d {entry['low_overlap_essays_vs_this_cohens_d'][L]:.2f}")
        print(f"{label}: {entry['pairs']} pairs, shared characters {entry['mean_shared_characters']:.2f}, "
              f"layer {L} difference {entry['difference'][L]:.3f} ± {entry['difference_sd'][L]:.3f}{extra}")
    low_entry = summary[essay_label]["low_overlap_quartile"]
    print(f"欽定四書文, lowest quarter of shared characters (<= {low_entry['max_shared_characters']:.2f}): "
          f"{low_entry['pairs']} pairs, layer {L} difference {low_entry['difference'][L]:.3f}")
    print("By period:", {k: (v["pairs"], round(v["difference"], 3)) for k, v in by_period.items()})
    print(f"Layer {L} difference by share of shared characters:")
    for label in records:
        print(f"  {label}: " + "  ".join(f"[{b['from']:.2f},{b['to']:.2f}) {b['difference']:.3f} ({b['pairs']})"
                                       if b["difference"] is not None else f"[{b['from']:.2f},{b['to']:.2f}) - (0)"
                                       for b in summary[label]["by_shared_characters_at_figure_layer"]))
    print(f"Essays with frame pairs (shared >= {args.frame_share}): {len(candidates)}; most: {candidates[:10]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.3, 1]})
    for label in records:
        axes[0].plot(range(layers), summary[label]["difference"], color=COLORS[label], linewidth=2,
                     label=f"{label} ({summary[label]['pairs']} pairs)")
    axes[0].plot(range(layers), low_entry["difference"], color=COLORS[essay_label], linewidth=1.5, linestyle="--",
                 label=f"{essay_label}, lowest quarter of shared characters")
    axes[0].axhline(0, color="0.6", linewidth=0.6)
    axes[0].set_xlabel("GujiBERT layer")
    axes[0].set_ylabel("Alignment minus random-pair baseline")
    axes[0].set_title("(a) Formally parallel pairs, by layer", fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    for position, label in enumerate(records):
        values = diff[label][:, L]
        jitter = np.random.default_rng(position).uniform(-0.2, 0.2, len(values))
        axes[1].scatter(position + jitter, values, s=6, color=COLORS[label], alpha=0.35, linewidths=0)
        axes[1].hlines(values.mean(), position - 0.3, position + 0.3, color="black", linewidth=1.5)
    axes[1].axhline(0, color="0.6", linewidth=0.6)
    axes[1].set_xticks(range(len(records)))
    axes[1].set_xticklabels(list(records))
    axes[1].set_title(f"(b) Every pair at layer {L}; bar = mean", fontsize=10, loc="left")
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "detected_pairs.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    (output_dir / "summary.json").write_text(json.dumps({
        "model": args.model, "rule": {"min_n": args.min_n, "max_n": args.max_n, "mean_gap": args.mean_gap, "max_gap": args.max_gap},
        "hand_marked_found": found, "hand_marked_total": total, "summary": summary,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "candidate_essays.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=1) + "\n",
                                                      encoding="utf-8")
    (output_dir / "pairs.json").write_text(json.dumps({label: [
        {"text": r["text"], "a": r["a"], "b": r["b"], "shared": round(r["shared"], 3),
         "alignment": np.round(r["alignment"], 4).tolist(), "baseline": np.round(r["baseline"], 4).tolist()}
        for r in steps] for label, steps in records.items()}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
