"""
Label-conditioned n-gram baseline: a count-based analogue of GujiGPT's I/O
control token.

The unlabeled combined model of ngram_baseline.py (line-internal Witten-Bell
trigram + position-matched cross-line p(c2_k | c1_k) + unigram + uniform) is
given label-specific components. Each component is estimated three times,
from all training couplets ("shared"), from the inner couplets only, and from
the outer couplets only. The I model mixes, for each component type, the
inner-count estimate with the shared estimate,

    p_I = sum_c w_c * [lambda_c * p_c^inner + (1 - lambda_c) * p_c^shared] + w_0 / V,

and the O model does the same with the outer counts. This works like
fine-tuning from a shared base. Because the nested mixture is linear, it is
fitted as one flat mixture by EM, and lambda_c is read off as
w(c, label) / (w(c, label) + w(c, shared)). The I model is tuned on the
inner held-out couplets and the O model on the outer held-out couplets, using
the same 10% of training poems as ngram_baseline.py (components fitted on the
other 90%), and both are then refitted on the full training split. The test
split is never used for tuning.

Every test couplet is scored under both models, as GujiGPT is scored under
both prompts. The script reports a Table-1-shaped summary (meter x couplet
type x prompt -> mean loss and entropy), prompt effects (matched minus
mismatched loss and entropy) for each couplet type, the I-minus-O entropy
difference within each type, and the per-position entropy gap between the
I-prompt inner and the O-prompt outer couplets. Tests are poem-paired
sign-flip tests (each poem's couplet differences averaged first); the
couplet-paired sign-flip test of decoding_io.py is reported alongside.

As a sparsity check, the whole procedure is repeated with shuffled labels:
within each training poem, two of the four couplets are chosen at random as
pseudo-inner, so the pseudo-I and pseudo-O models are trained on random halves
of the same size as the real label halves. Test couplets keep their true
labels, so any prompt effect in this run comes from smoothing or chance alone.
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict

import numpy as np

from common import METER_NAMES
from ngram_baseline import UNK, Components, em_weights, iter_targets, load_couplets

METERS = (5, 7)
DEFAULT_OUT_DIR = "results/ngram_baseline"
GUJIGPT_PATH = "results/gpt_decoding/decoding_io_gujigpt_io_{run}_{meter}_aligned_by_position_stats.json"
ROWS = (  # dense component rows used for scoring
    "tri_I", "tri_O", "tri_S",
    "cross_I", "cross_O", "cross_S",
    "uni_I", "uni_O", "uni_S",
    "uniform",
)
PROMPT_COMPONENTS = {
    p: (f"tri_{p}", "tri_S", f"cross_{p}", "cross_S", f"uni_{p}", "uni_S", "uniform")
    for p in ("I", "O")
}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_components(records: list[dict], vocab: dict, label_key: str) -> dict:
    """Shared, inner-only, and outer-only count components."""
    return {
        "S": Components(records, vocab),
        "I": Components([r for r in records if r[label_key]], vocab),
        "O": Components([r for r in records if not r[label_key]], vocab),
    }


def target_probs(comps: dict, records: list[dict], vocab: dict) -> dict:
    """Probability of the true character under every component, for EM."""
    out = {name: [] for name in ROWS}
    V = comps["S"].V
    for _, _, p2, p1, c1, c in iter_targets(records, vocab):
        for key in ("I", "O", "S"):
            cm = comps[key]
            out[f"tri_{key}"].append(cm.prob_of(cm.line_internal(p2, p1), c))
            out[f"cross_{key}"].append(cm.prob_of(cm.cross_line(c1), c))
            out[f"uni_{key}"].append(cm.p_uni[c])
        out["uniform"].append(1.0 / V)
    return {k: np.asarray(v) for k, v in out.items()}


def tune(train: list[dict], vocab: dict, label_key: str, heldout_poems: set) -> dict:
    """EM-tune the I and O mixtures on held-out training couplets of each label."""
    fit = [r for r in train if r["poem_id"] not in heldout_poems]
    heldout = [r for r in train if r["poem_id"] in heldout_poems]
    comps = fit_components(fit, vocab, label_key)
    tuned = {}
    for prompt in ("I", "O"):
        want = prompt == "I"
        recs = [r for r in heldout if r[label_key] == want]
        probs = target_probs(comps, recs, vocab)
        res = em_weights(probs, PROMPT_COMPONENTS[prompt])
        w = res["weights"]
        res["lambda"] = {
            kind: w[f"{kind}_{prompt}"] / (w[f"{kind}_{prompt}"] + w[f"{kind}_S"])
            if (w[f"{kind}_{prompt}"] + w[f"{kind}_S"]) > 0
            else float("nan")
            for kind in ("tri", "cross", "uni")
        }
        res["component_type_weight"] = {
            kind: w[f"{kind}_{prompt}"] + w[f"{kind}_S"] for kind in ("tri", "cross", "uni")
        }
        res["component_type_weight"]["uniform"] = w["uniform"]
        res["n_heldout_couplets"] = len(recs)
        # Held-out loss of the shared-only mixture, refitted on the same couplets,
        # shows how much the label-specific counts help before any test scoring.
        shared_only = em_weights(probs, ("tri_S", "cross_S", "uni_S", "uniform"))
        res["heldout_loss_shared_only"] = shared_only["heldout_loss"]
        tuned[prompt] = res
    return tuned


def score(comps: dict, records: list[dict], vocab: dict, tuned: dict) -> dict:
    """Loss and full-distribution entropy for every test target under both prompts."""
    V = comps["S"].V
    uniform = np.full(V, 1.0 / V)
    W = np.zeros((2, len(ROWS)))
    for i, prompt in enumerate(("I", "O")):
        for name, w in tuned[prompt]["weights"].items():
            W[i, ROWS.index(name)] = w
    meter = len(records[0]["line2"])
    n = len(records) * meter
    loss = np.empty((2, n))
    ent = np.empty((2, n))
    max_err = 0.0
    D = np.empty((len(ROWS), V))
    for t, (_, _, p2, p1, c1, c) in enumerate(iter_targets(records, vocab)):
        for j, key in enumerate(("I", "O", "S")):
            cm = comps[key]
            D[j] = cm.dense(cm.line_internal(p2, p1))
            D[3 + j] = cm.dense(cm.cross_line(c1))
            D[6 + j] = cm.p_uni
        D[9] = uniform
        P = W @ D
        max_err = max(max_err, float(np.abs(P.sum(axis=1) - 1.0).max()))
        loss[:, t] = -np.log(P[:, c])
        ent[:, t] = -(P * np.log(P)).sum(axis=1)
    return {
        "loss": {p: loss[i].reshape(len(records), meter) for i, p in enumerate("IO")},
        "entropy": {p: ent[i].reshape(len(records), meter) for i, p in enumerate("IO")},
        "max_prob_sum_error": max_err,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def sign_flip_p(diffs: np.ndarray, rng, n_perm: int) -> float:
    """Two-sided sign-flip permutation p-value for mean(diffs) = 0."""
    observed = abs(float(diffs.mean()))
    extreme = 0
    for start in range(0, n_perm, 500):
        m = min(500, n_perm - start)
        signs = rng.integers(0, 2, size=(m, len(diffs))) * 2 - 1
        extreme += int((np.abs((signs * diffs).mean(axis=1)) >= observed - 1e-12).sum())
    return (extreme + 1) / (n_perm + 1)


def poem_means(values: np.ndarray, poem_ids: list[str]) -> dict:
    """Average per-couplet values within each poem."""
    acc = defaultdict(list)
    for v, pid in zip(values, poem_ids):
        acc[pid].append(v)
    return {pid: float(np.mean(v)) for pid, v in acc.items()}


def paired_test(diffs: np.ndarray, poem_ids: list[str], rng, n_perm: int) -> dict:
    """Mean of per-couplet paired differences with poem- and couplet-level sign-flip p-values."""
    by_poem = np.asarray(list(poem_means(diffs, poem_ids).values()))
    return {
        "mean": float(diffs.mean()),
        "n_couplets": int(len(diffs)),
        "n_poems": int(len(by_poem)),
        "poem_sign_flip_p": sign_flip_p(by_poem, rng, n_perm),
        "couplet_sign_flip_p": sign_flip_p(diffs, rng, n_perm),
    }


def summarize(scores: dict, records: list[dict], rng, n_perm: int) -> dict:
    """Table-1 cells, prompt effects, I-vs-O entropy differences, per-position gaps."""
    inner = np.asarray([r["inner"] for r in records])
    pids = np.asarray([r["poem_id"] for r in records])
    out = {"table1": [], "prompt_effects": {}, "entropy_I_minus_O": {}, "per_position": []}
    for ctype, mask in (("inner", inner), ("outer", ~inner)):
        matched = "I" if ctype == "inner" else "O"
        mismatched = "O" if matched == "I" else "I"
        for prompt in (matched, mismatched):
            out["table1"].append(
                {
                    "couplet_type": ctype,
                    "prompt": prompt,
                    "condition": "matched" if prompt == matched else "mismatched",
                    "n_couplets": int(mask.sum()),
                    "mean_loss": float(scores["loss"][prompt][mask].mean()),
                    "mean_entropy": float(scores["entropy"][prompt][mask].mean()),
                }
            )
        eff = {}
        for metric in ("loss", "entropy"):
            d = (scores[metric][matched][mask] - scores[metric][mismatched][mask]).mean(axis=1)
            eff[metric] = paired_test(d, list(pids[mask]), rng, n_perm)
        out["prompt_effects"][ctype] = eff
        d = (scores["entropy"]["I"][mask] - scores["entropy"]["O"][mask]).mean(axis=1)
        out["entropy_I_minus_O"][ctype] = paired_test(d, list(pids[mask]), rng, n_perm)

    # Aligned gap: I-prompt inner minus O-prompt outer, paired by poem.
    meter = scores["entropy"]["I"].shape[1]
    for k in range(-1, meter):
        row = {}
        for metric in ("entropy", "loss"):
            a = scores[metric]["I"][inner]
            b = scores[metric]["O"][~inner]
            a = a.mean(axis=1) if k < 0 else a[:, k]
            b = b.mean(axis=1) if k < 0 else b[:, k]
            ma, mb = poem_means(a, list(pids[inner])), poem_means(b, list(pids[~inner]))
            shared = sorted(set(ma) & set(mb))
            diffs = np.asarray([ma[p] - mb[p] for p in shared])
            row[metric] = {
                "inner_I": float(a.mean()),
                "outer_O": float(b.mean()),
                "diff": float(a.mean() - b.mean()),
                "poem_sign_flip_p": sign_flip_p(diffs, rng, n_perm),
            }
        if k < 0:
            out["aligned_whole_line"] = row
        else:
            row["char_position"] = k + 1
            out["per_position"].append(row)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_condition(train, test, vocab, heldout_poems, label_key, rng, n_perm) -> dict:
    """Tune, refit, score, and summarize for one labelling of the training data."""
    tuned = tune(train, vocab, label_key, heldout_poems)
    comps = fit_components(train, vocab, label_key)
    scores = score(comps, test, vocab, tuned)
    return {
        "tuning": tuned,
        "max_prob_sum_error": scores["max_prob_sum_error"],
        "results": summarize(scores, test, rng, n_perm),
    }


def gujigpt_reference(meter_name: str) -> dict:
    """Matched/mismatched rows and tests of the fine-tuned and shuffled-label GujiGPT runs."""
    ref = {}
    for run in ("finetuned", "shuffled"):
        path = GUJIGPT_PATH.format(run=run, meter=meter_name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            ref[run] = {
                "rows": [
                    {k: r[k] for k in ("actual_couplet_type", "prompt", "mean_loss", "mean_entropy")}
                    for r in d["matched_mismatched_rows"]
                ],
                "tests": [
                    {
                        k: t[k]
                        for k in (
                            "actual_couplet_type",
                            "mean_loss_diff_matched_minus_mismatched",
                            "mean_entropy_diff_matched_minus_mismatched",
                        )
                    }
                    for t in d["matched_mismatched_tests"]
                ],
            }
    return ref


def run_meter(meter: int, args, rng) -> dict:
    name = METER_NAMES[meter]
    train, skipped_train = load_couplets(f"data/2-3_regulated_verse/couplets_{name}_train.json", meter)
    test, skipped_test = load_couplets(f"data/2-3_regulated_verse/couplets_{name}_test.json", meter)

    chars = sorted({c for r in train for c in r["line1"] + r["line2"]})
    vocab = {c: i for i, c in enumerate(chars)}
    vocab[UNK] = len(vocab)

    # Same held-out poems as ngram_baseline.py.
    poems = sorted({r["poem_id"] for r in train})
    random.Random(args.seed).shuffle(poems)
    heldout_poems = set(poems[: int(round(len(poems) * args.heldout_fraction))])

    # Shuffled labels: two random couplets of each training poem become pseudo-inner.
    shuffle_rng = random.Random(args.seed + 1)
    by_poem = defaultdict(list)
    for r in train:
        by_poem[r["poem_id"]].append(r)
    for pid in sorted(by_poem):
        recs = by_poem[pid]
        chosen = set(shuffle_rng.sample(range(len(recs)), len(recs) // 2))
        for i, r in enumerate(recs):
            r["pseudo_inner"] = i in chosen
    agreement = float(np.mean([r["pseudo_inner"] == r["inner"] for r in train]))

    print(f"  {name}: true labels", flush=True)
    true_run = run_condition(train, test, vocab, heldout_poems, "inner", rng, args.num_permutations)
    print(f"  {name}: shuffled labels", flush=True)
    shuf_run = run_condition(
        train, test, vocab, heldout_poems, "pseudo_inner", rng, args.num_permutations
    )

    unlabeled = None
    path = os.path.join(args.output_dir, "ngram_baseline.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            comb = json.load(f)["meters"][name]["results"]["combined"]
        unlabeled = {
            m: {"inner": comb[m]["inner"], "outer": comb[m]["outer"]} for m in ("loss", "entropy")
        }

    return {
        "meter": meter,
        "counts": {
            "train_couplets": len(train),
            "train_inner": int(sum(r["inner"] for r in train)),
            "train_outer": int(sum(not r["inner"] for r in train)),
            "train_skipped": dict(skipped_train),
            "test_couplets": len(test),
            "test_skipped": dict(skipped_test),
            "test_poems": len({r["poem_id"] for r in test}),
            "heldout_poems_for_tuning": len(heldout_poems),
        },
        "vocab_size_including_unk": len(vocab),
        "shuffled_label_agreement_with_true": agreement,
        "true_labels": true_run,
        "shuffled_labels": shuf_run,
        "unlabeled_combined_reference": unlabeled,
        "gujigpt_reference": gujigpt_reference(name),
    }


def fmt_p(p: float) -> str:
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def write_summary(all_results: dict, path: str, n_perm: int) -> None:
    L = [
        "# Label-conditioned n-gram baseline (count analogue of the I/O control token)",
        "",
        "Each component (trigram, cross-line, unigram) has inner-only and outer-only count estimates "
        "interpolated with the shared all-couplet estimate; weights and lambda are tuned by EM on 10% "
        "of the training poems with true labels. Every test couplet is scored under both the I and the "
        "O model. Values are nats per second-line character. p is a poem-paired sign-flip test "
        f"({n_perm:,} permutations); p(couplet) is the couplet-paired test of decoding_io.py.",
        "",
    ]
    for name, res in all_results.items():
        meter = res["meter"]
        L += [f"## {name} ({meter}-character lines)", ""]
        for cond, title in (("true_labels", "True labels"), ("shuffled_labels", "Shuffled labels (sparsity check)")):
            r = res[cond]["results"]
            L += [f"### {title}", "", "| Couplet | Prompt | Condition | Loss | Entropy |", "|---|---|---|---|---|"]
            for row in r["table1"]:
                L.append(
                    f"| {row['couplet_type']} | {row['prompt']} | {row['condition']} | "
                    f"{row['mean_loss']:.3f} | {row['mean_entropy']:.3f} |"
                )
            L += ["", "| Couplet | Loss matched−mismatched | p | p(couplet) | Entropy I−O | p | p(couplet) |", "|---|---|---|---|---|---|---|"]
            for ctype in ("inner", "outer"):
                pe = r["prompt_effects"][ctype]["loss"]
                ei = r["entropy_I_minus_O"][ctype]
                L.append(
                    f"| {ctype} | {pe['mean']:+.3f} | {fmt_p(pe['poem_sign_flip_p'])} | "
                    f"{fmt_p(pe['couplet_sign_flip_p'])} | {ei['mean']:+.3f} | "
                    f"{fmt_p(ei['poem_sign_flip_p'])} | {fmt_p(ei['couplet_sign_flip_p'])} |"
                )
            L += ["", "Entropy gap, I-prompt inner minus O-prompt outer, by second-line position:", ""]
            L.append("| Whole line | " + " | ".join(str(k) for k in range(1, meter + 1)) + " |")
            L.append("|---|" + "---|" * meter)
            aw = r["aligned_whole_line"]["entropy"]
            L.append(
                f"| {aw['diff']:+.3f} | "
                + " | ".join(
                    f"{p['entropy']['diff']:+.3f}{'*' if p['entropy']['poem_sign_flip_p'] < 0.05 else ''}"
                    for p in r["per_position"]
                )
                + " |"
            )
            L += ["", "Tuned weights (component type total, lambda = label-specific share):", ""]
            for prompt, t in res[cond]["tuning"].items():
                cw, lam = t["component_type_weight"], t["lambda"]
                L.append(
                    f"- {prompt} model: trigram {cw['tri']:.3f} (lambda {lam['tri']:.3f}), "
                    f"cross {cw['cross']:.3f} (lambda {lam['cross']:.3f}), "
                    f"unigram {cw['uni']:.3f} (lambda {lam['uni']:.3f}), uniform {cw['uniform']:.4f}; "
                    f"held-out loss {t['heldout_loss']:.3f} vs shared-only {t['heldout_loss_shared_only']:.3f}"
                )
            L.append("")
        u = res.get("unlabeled_combined_reference")
        if u:
            L.append(
                f"Unlabeled combined model (ngram_baseline.py): loss inner {u['loss']['inner']:.3f}, "
                f"outer {u['loss']['outer']:.3f}; entropy inner {u['entropy']['inner']:.3f}, "
                f"outer {u['entropy']['outer']:.3f}."
            )
            L.append("")
        g = res.get("gujigpt_reference", {}).get("finetuned")
        if g:
            L += ["GujiGPT fine-tuned reference (decoding_io.py):", "", "| Couplet | Prompt | Loss | Entropy |", "|---|---|---|---|"]
            for row in g["rows"]:
                L.append(
                    f"| {row['actual_couplet_type']} | {row['prompt']} | {row['mean_loss']:.3f} | {row['mean_entropy']:.3f} |"
                )
            L.append("")
        L.append("* poem-paired p < .05.")
        L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-permutations", type=int, default=10_000)
    parser.add_argument("--heldout-fraction", type=float, default=0.10)
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_results = {}
    for meter in METERS:
        print(f"Running {METER_NAMES[meter]}...", flush=True)
        all_results[METER_NAMES[meter]] = run_meter(meter, args, rng)

    payload = {
        "description": __doc__.strip(),
        "seed": args.seed,
        "num_permutations": args.num_permutations,
        "heldout_fraction": args.heldout_fraction,
        "meters": all_results,
    }
    with open(os.path.join(args.output_dir, "labeled.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    out_md = os.path.join(args.output_dir, "labeled_summary.md")
    write_summary(all_results, out_md, args.num_permutations)
    print(open(out_md, encoding="utf-8").read())


if __name__ == "__main__":
    main()
