"""
Model-free lexical baseline for the inner/outer couplet contrast.

For each meter, the script fits plain count-based character models on the
training couplets (inner/outer labels unused, one model for all couplets of a
meter) and scores every character of each test couplet's second line given
the first line and the preceding second-line characters, as the neural models
are scored in decoding_io.py. Three models are reported:

1. line_internal: a Witten-Bell interpolated character trigram over the
   second line (trigram -> bigram -> unigram -> uniform), with line-start
   padding, mixed with an explicit uniform floor.
2. cross_line: a position-matched cross-line model p(c2_k | c1_k), Witten-Bell
   smoothed toward the unigram, mixed with the unigram and a uniform floor.
3. combined: a linear mixture of the line-internal model, the cross-line
   model, the unigram, and the uniform floor. This is the main baseline.

A context-free unigram (plus uniform floor) is also reported as a reference.

Mixture weights are tuned by EM on held-out likelihood over 10% of the
training poems (components fitted on the other 90%), then all components are
refitted on the full training split with those weights. The test split is
never used for tuning. The vocabulary is every character in the training
couplets plus an UNK bucket, and every predictive distribution is a proper
distribution over that vocabulary (checked numerically), so unseen
characters get finite loss through the uniform floor.

Loss is the negative natural log probability of the true character (nats),
and entropy is the entropy of the full predictive distribution (nats).
Inner/outer differences are tested with (a) a couplet-level label
permutation test and (b) the poem-paired sign-flip test used for GujiGPT in
decoding_io.py. Text is used as stored (simplified Chinese); OpenCC
conversion is unnecessary because train and test are in the same script.
"""

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np

from common import METER_NAMES, has_junk_characters, io_label_for_position, split_couplet

METERS = (5, 7)
BOS = "<s>"
UNK = "<unk>"
DEFAULT_OUT_DIR = "results/ngram_baseline"
GUJIGPT_BASE_PATH = (
    "results/gpt_decoding/decoding_io_gujigpt_base_no_prompt_{meter}_aligned_by_position_stats.json"
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_couplets(path: str, meter: int) -> tuple[list[dict], Counter]:
    """Read couplet records, keeping only well-formed ones and counting the rest."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    kept, skipped = [], Counter()
    for item in data:
        position = item.get("position")
        if position not in (1, 2, 3, 4):
            skipped["bad_position"] += 1
            continue
        line1, line2 = split_couplet(item.get("couplet", ""))
        if line1 is None:
            skipped["no_comma_split"] += 1
            continue
        if len(line1) != meter or len(line2) != meter:
            skipped["wrong_length"] += 1
            continue
        if has_junk_characters(line1 + line2):
            skipped["junk_characters"] += 1
            continue
        label = io_label_for_position(position)
        if item.get("label") != label:
            skipped["label_position_mismatch"] += 1
            continue
        kept.append(
            {
                "line1": line1,
                "line2": line2,
                "inner": label == "i",
                "position": position,
                "poem_id": item["poem_id"],
            }
        )
    return kept, skipped


# ---------------------------------------------------------------------------
# Count models
# ---------------------------------------------------------------------------


class Components:
    """Count-based component distributions fitted on a set of couplets."""

    def __init__(self, records: list[dict], vocab: dict[str, int]):
        self.V = len(vocab)
        self.vocab = vocab
        unk = vocab[UNK]
        self.unk = unk
        uni = np.zeros(self.V)
        bi = defaultdict(Counter)
        tri = defaultdict(Counter)
        cross = defaultdict(Counter)
        for rec in records:
            l1 = [vocab.get(c, unk) for c in rec["line1"]]
            l2 = [vocab.get(c, unk) for c in rec["line2"]]
            hist = [-1, -1] + l2  # -1 is the line-start symbol
            for k, c in enumerate(l2):
                uni[c] += 1
                bi[hist[k + 1]][c] += 1
                tri[(hist[k], hist[k + 1])][c] += 1
                cross[l1[k]][c] += 1
        # Witten-Bell unigram toward uniform.
        n_tokens = uni.sum()
        n_types = float((uni > 0).sum())
        self.p_uni = (uni + n_types / self.V) / (n_tokens + n_types)
        self.bi = self._freeze(bi)
        self.tri = self._freeze(tri)
        self.cross = self._freeze(cross)

    @staticmethod
    def _freeze(table: dict) -> dict:
        """Store each context as (indices, counts, total, types) arrays."""
        frozen = {}
        for ctx, counter in table.items():
            idx = np.fromiter(counter.keys(), dtype=np.int64, count=len(counter))
            cnt = np.fromiter(counter.values(), dtype=float, count=len(counter))
            frozen[ctx] = (idx, cnt, float(cnt.sum()), float(len(cnt)))
        return frozen

    # Each component is represented as  a * p_uni + sparse(idx, val),
    # which is a proper distribution because a + sum(val) = 1.

    def _wb_step(self, entry, lower):
        """Witten-Bell step: combine a context's counts with a lower-order (a, idx, val)."""
        a_low, idx_low, val_low = lower
        if entry is None:
            return lower
        idx, cnt, total, types = entry
        denom = total + types
        w = types / denom
        return (
            w * a_low,
            np.concatenate([idx, idx_low]),
            np.concatenate([cnt / denom, w * val_low]),
        )

    def line_internal(self, prev2: int, prev1: int):
        """Witten-Bell trigram -> bigram -> unigram for the next second-line character."""
        base = (1.0, np.empty(0, dtype=np.int64), np.empty(0))
        p_bi = self._wb_step(self.bi.get(prev1), base)
        return self._wb_step(self.tri.get((prev2, prev1)), p_bi)

    def cross_line(self, c1: int):
        """Witten-Bell p(c2_k | c1_k) toward the unigram."""
        base = (1.0, np.empty(0, dtype=np.int64), np.empty(0))
        return self._wb_step(self.cross.get(c1), base)

    def prob_of(self, comp, target: int) -> float:
        """Probability a sparse-plus-unigram component assigns to one character."""
        a, idx, val = comp
        return a * self.p_uni[target] + float(val[idx == target].sum())

    def dense(self, comp) -> np.ndarray:
        """Expand a component to a dense probability vector."""
        a, idx, val = comp
        vec = a * self.p_uni
        np.add.at(vec, idx, val)
        return vec


MODELS = {
    # model name -> components in the mixture; "unigram" is a context-free
    # reference showing how the two second-line vocabularies differ on their own.
    "unigram": ("unigram", "uniform"),
    "line_internal": ("tri", "uniform"),
    "cross_line": ("cross", "unigram", "uniform"),
    "combined": ("tri", "cross", "unigram", "uniform"),
}


def iter_targets(records: list[dict], vocab: dict[str, int]):
    """Yield (record index, char position k, prev2, prev1, c1_k, target) for every second-line character."""
    unk = vocab[UNK]
    for r, rec in enumerate(records):
        l1 = [vocab.get(c, unk) for c in rec["line1"]]
        l2 = [vocab.get(c, unk) for c in rec["line2"]]
        hist = [-1, -1] + l2
        for k, c in enumerate(l2):
            yield r, k, hist[k], hist[k + 1], l1[k], c


def component_target_probs(comps: Components, records: list[dict], vocab: dict) -> dict:
    """Probability of the true character under each component, for EM tuning."""
    out = {"tri": [], "cross": [], "unigram": [], "uniform": []}
    for _, _, p2, p1, c1, c in iter_targets(records, vocab):
        out["tri"].append(comps.prob_of(comps.line_internal(p2, p1), c))
        out["cross"].append(comps.prob_of(comps.cross_line(c1), c))
        out["unigram"].append(comps.p_uni[c])
        out["uniform"].append(1.0 / comps.V)
    return {k: np.asarray(v) for k, v in out.items()}


def em_weights(probs: dict, names: tuple, iters: int = 500, tol: float = 1e-10) -> dict:
    """Maximum held-out-likelihood mixture weights by EM."""
    P = np.stack([probs[n] for n in names], axis=1)
    w = np.full(len(names), 1.0 / len(names))
    prev = -np.inf
    for _ in range(iters):
        mix = P @ w
        ll = float(np.log(mix).mean())
        resp = P * w / mix[:, None]
        w = resp.mean(axis=0)
        if ll - prev < tol:
            break
        prev = ll
    mix = P @ w
    return {
        "weights": {n: float(x) for n, x in zip(names, w)},
        "heldout_loss": float(-np.log(mix).mean()),
    }


def score_test(comps: Components, records: list[dict], vocab: dict, weights: dict) -> dict:
    """Per-target loss and full-distribution entropy for every model."""
    V = comps.V
    uniform = np.full(V, 1.0 / V)
    n = sum(len(r["line2"]) for r in records)
    meter = len(records[0]["line2"])
    out = {m: {"loss": np.empty(n), "entropy": np.empty(n)} for m in MODELS}
    max_sum_err = 0.0
    for t, (_, _, p2, p1, c1, c) in enumerate(iter_targets(records, vocab)):
        dense = {
            "tri": comps.dense(comps.line_internal(p2, p1)),
            "cross": comps.dense(comps.cross_line(c1)),
            "unigram": comps.p_uni,
            "uniform": uniform,
        }
        for model, names in MODELS.items():
            w = weights[model]["weights"]
            p = sum(w[nm] * dense[nm] for nm in names)
            max_sum_err = max(max_sum_err, abs(p.sum() - 1.0))
            out[model]["loss"][t] = -math.log(p[c])
            out[model]["entropy"][t] = float(-(p * np.log(p)).sum())
    for model in out:
        for metric in ("loss", "entropy"):
            out[model][metric] = out[model][metric].reshape(len(records), meter)
    out["_max_prob_sum_error"] = max_sum_err
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def label_permutation_p(values: np.ndarray, inner: np.ndarray, rng, n_perm: int) -> float:
    """Two-sided couplet-level label permutation p-value for mean(inner) - mean(outer)."""
    n_in = int(inner.sum())
    total = values.sum()
    observed = values[inner].mean() - values[~inner].mean()
    extreme = 0
    for start in range(0, n_perm, 500):
        m = min(500, n_perm - start)
        perm = rng.permuted(np.tile(inner, (m, 1)), axis=1)
        s_in = (perm * values).sum(axis=1)
        diff = s_in / n_in - (total - s_in) / (len(values) - n_in)
        extreme += int((np.abs(diff) >= abs(observed) - 1e-12).sum())
    return (extreme + 1) / (n_perm + 1)


def poem_paired(values: np.ndarray, records: list[dict], inner: np.ndarray, rng, n_perm: int) -> dict:
    """Poem-paired inner-minus-outer difference with a sign-flip p-value (as in decoding_io.py)."""
    by_poem = defaultdict(lambda: {True: [], False: []})
    for v, rec, is_in in zip(values, records, inner):
        by_poem[rec["poem_id"]][bool(is_in)].append(v)
    diffs = np.asarray(
        [np.mean(d[True]) - np.mean(d[False]) for d in by_poem.values() if d[True] and d[False]]
    )
    observed = abs(diffs.mean())
    extreme = 0
    for start in range(0, n_perm, 500):
        m = min(500, n_perm - start)
        signs = rng.integers(0, 2, size=(m, len(diffs))) * 2 - 1
        extreme += int((np.abs((signs * diffs).mean(axis=1)) >= observed - 1e-12).sum())
    return {
        "n_poems": int(len(diffs)),
        "mean_difference": float(diffs.mean()),
        "sign_flip_p": (extreme + 1) / (n_perm + 1),
    }


def summarize(scores: dict, records: list[dict], rng, n_perm: int) -> dict:
    """Inner/outer means, differences, tests, per-position and per-couplet-slot breakdowns."""
    inner = np.asarray([r["inner"] for r in records])
    slots = np.asarray([r["position"] for r in records])
    out = {}
    for model in MODELS:
        res = {}
        for metric in ("loss", "entropy"):
            mat = scores[model][metric]
            per_couplet = mat.mean(axis=1)
            mi, mo = per_couplet[inner].mean(), per_couplet[~inner].mean()
            per_pos = []
            for k in range(mat.shape[1]):
                col = mat[:, k]
                per_pos.append(
                    {
                        "char_position": k + 1,
                        "inner": float(col[inner].mean()),
                        "outer": float(col[~inner].mean()),
                        "diff": float(col[inner].mean() - col[~inner].mean()),
                        "label_perm_p": label_permutation_p(col, inner, rng, n_perm),
                    }
                )
            res[metric] = {
                "inner": float(mi),
                "outer": float(mo),
                "diff_inner_minus_outer": float(mi - mo),
                "label_perm_p": label_permutation_p(per_couplet, inner, rng, n_perm),
                "poem_paired": poem_paired(per_couplet, records, inner, rng, n_perm),
                "by_char_position": per_pos,
                "by_couplet_slot": {
                    int(s): float(per_couplet[slots == s].mean()) for s in (1, 2, 3, 4)
                },
            }
        out[model] = res
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_meter(meter: int, args, rng_np) -> dict:
    """Fit, tune, and evaluate the baseline for one meter."""
    name = METER_NAMES[meter]
    train, skipped_train = load_couplets(f"data/2-3_regulated_verse/couplets_{name}_train.json", meter)
    test, skipped_test = load_couplets(f"data/2-3_regulated_verse/couplets_{name}_test.json", meter)

    chars = sorted({c for r in train for c in r["line1"] + r["line2"]})
    vocab = {c: i for i, c in enumerate(chars)}
    vocab[UNK] = len(vocab)

    test_chars = [c for r in test for c in r["line2"]]
    test_chars_all = [c for r in test for c in r["line1"] + r["line2"]]
    oov_line2 = sum(c not in vocab for c in test_chars) / len(test_chars)
    oov_all = sum(c not in vocab for c in test_chars_all) / len(test_chars_all)

    # Held-out slice of training poems for weight tuning.
    poems = sorted({r["poem_id"] for r in train})
    random.Random(args.seed).shuffle(poems)
    heldout_poems = set(poems[: int(round(len(poems) * args.heldout_fraction))])
    fit = [r for r in train if r["poem_id"] not in heldout_poems]
    heldout = [r for r in train if r["poem_id"] in heldout_poems]

    tune_comps = Components(fit, vocab)
    probs = component_target_probs(tune_comps, heldout, vocab)
    weights = {model: em_weights(probs, names) for model, names in MODELS.items()}

    comps = Components(train, vocab)
    scores = score_test(comps, test, vocab, weights)
    stats = summarize(scores, test, rng_np, args.num_permutations)

    gujigpt = None
    path = GUJIGPT_BASE_PATH.format(meter=name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            gujigpt = json.load(f).get("no_prompt_type_comparison", {}).get("metrics")

    return {
        "meter": meter,
        "counts": {
            "train_couplets": len(train),
            "train_skipped": dict(skipped_train),
            "test_couplets": len(test),
            "test_skipped": dict(skipped_test),
            "test_inner": int(sum(r["inner"] for r in test)),
            "test_outer": int(sum(not r["inner"] for r in test)),
            "test_poems": len({r["poem_id"] for r in test}),
            "fit_couplets_for_tuning": len(fit),
            "heldout_couplets_for_tuning": len(heldout),
            "heldout_poems_for_tuning": len(heldout_poems),
            "scored_characters": int(scores["combined"]["loss"].size),
        },
        "vocab_size_including_unk": len(vocab),
        "oov_rate_test_second_line": oov_line2,
        "oov_rate_test_all_lines": oov_all,
        "max_probability_sum_error": scores["_max_prob_sum_error"],
        "interpolation": weights,
        "results": stats,
        "gujigpt_base_no_prompt_reference": gujigpt,
    }


def fmt_p(p: float) -> str:
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def write_summary(all_results: dict, path: str, n_perm: int) -> None:
    """Write a short Markdown table of the main numbers."""
    lines = [
        "# Model-free n-gram baseline: inner vs outer second lines",
        "",
        "Labels unused in training; weights tuned on 10% of training poems; test split only for scoring.",
        f"Loss and entropy in nats per second-line character. Diff is inner minus outer; "
        f"p is a couplet-level label permutation test ({n_perm:,} permutations), "
        "and p(poem) the poem-paired sign-flip test used for GujiGPT.",
        "",
    ]
    for name, res in all_results.items():
        c = res["counts"]
        lines += [
            f"## {name} ({res['meter']}-character lines)",
            "",
            f"Test couplets {c['test_couplets']} ({c['test_inner']} inner, {c['test_outer']} outer, "
            f"{c['test_poems']} poems); train couplets {c['train_couplets']}; skipped train "
            f"{sum(c['train_skipped'].values())}, test {sum(c['test_skipped'].values())}. "
            f"Vocabulary {res['vocab_size_including_unk']} (incl. UNK); OOV rate on test "
            f"second lines {res['oov_rate_test_second_line']:.4%}.",
            "",
            "| Model | Loss in | Loss out | Diff | p | p(poem) | Ent in | Ent out | Diff | p | p(poem) |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        rows = list(res["results"].items())
        for model, r in rows:
            L, E = r["loss"], r["entropy"]
            lines.append(
                f"| {model} | {L['inner']:.3f} | {L['outer']:.3f} | {L['diff_inner_minus_outer']:+.3f} | "
                f"{fmt_p(L['label_perm_p'])} | {fmt_p(L['poem_paired']['sign_flip_p'])} | "
                f"{E['inner']:.3f} | {E['outer']:.3f} | {E['diff_inner_minus_outer']:+.3f} | "
                f"{fmt_p(E['label_perm_p'])} | {fmt_p(E['poem_paired']['sign_flip_p'])} |"
            )
        g = res.get("gujigpt_base_no_prompt_reference")
        if g:
            lines.append(
                f"| GujiGPT base (reference) | {g['loss']['mean_inner']:.3f} | {g['loss']['mean_outer']:.3f} | "
                f"{g['loss']['mean_difference_inner_minus_outer']:+.3f} | | | "
                f"{g['entropy']['mean_inner']:.3f} | {g['entropy']['mean_outer']:.3f} | "
                f"{g['entropy']['mean_difference_inner_minus_outer']:+.3f} | | |"
            )
        lines += ["", "Inner-minus-outer entropy gap by second-line position (nats):", ""]
        meter = res["meter"]
        lines.append("| Model | " + " | ".join(str(k) for k in range(1, meter + 1)) + " |")
        lines.append("|---|" + "---|" * meter)
        for model, r in rows:
            lines.append(
                f"| {model} | "
                + " | ".join(f"{p['diff']:+.3f}" for p in r["entropy"]["by_char_position"])
                + " |"
            )
        lines += ["", "Inner-minus-outer loss gap by second-line position (nats):", ""]
        lines.append("| Model | " + " | ".join(str(k) for k in range(1, meter + 1)) + " |")
        lines.append("|---|" + "---|" * meter)
        for model, r in rows:
            lines.append(
                f"| {model} | "
                + " | ".join(f"{p['diff']:+.3f}" for p in r["loss"]["by_char_position"])
                + " |"
            )
        lines += ["", "Combined-model mean loss by couplet slot (1 and 4 outer, 2 and 3 inner):", ""]
        slots = res["results"]["combined"]["loss"]["by_couplet_slot"]
        lines.append("| Slot 1 | Slot 2 | Slot 3 | Slot 4 |")
        lines.append("|---|---|---|---|")
        lines.append("| " + " | ".join(f"{slots[s]:.3f}" for s in (1, 2, 3, 4)) + " |")
        lines += ["", "Tuned interpolation weights (held-out loss):", ""]
        for model, w in res["interpolation"].items():
            ws = ", ".join(f"{k} {v:.4f}" for k, v in w["weights"].items())
            lines.append(f"- {model}: {ws} (held-out loss {w['heldout_loss']:.3f})")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-permutations", type=int, default=10_000)
    parser.add_argument("--heldout-fraction", type=float, default=0.10)
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng_np = np.random.default_rng(args.seed)
    all_results = {}
    for meter in METERS:
        print(f"Running {METER_NAMES[meter]}...", flush=True)
        all_results[METER_NAMES[meter]] = run_meter(meter, args, rng_np)

    payload = {
        "description": __doc__.strip(),
        "seed": args.seed,
        "num_permutations": args.num_permutations,
        "heldout_fraction": args.heldout_fraction,
        "meters": all_results,
    }
    with open(os.path.join(args.output_dir, "ngram_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    write_summary(all_results, os.path.join(args.output_dir, "summary.md"), args.num_permutations)
    print(open(os.path.join(args.output_dir, "summary.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
