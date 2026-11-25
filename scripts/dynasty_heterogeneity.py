"""Check whether the inner/outer contrast differs between dynasties.

Uses the 1,000 inner and 1,000 outer test couplets per meter of the positional
analysis (200 per dynasty), joined to their dynasty through
data/2-3_regulated_verse/couplets_{meter}_test.json. For each dynasty it computes the inner/outer
gap in mean next-character entropy and loss (fine-tuned GujiGPT, aligned
prompts) and in transition alignment (GujiBERT layer 8). Heterogeneity across
dynasties is tested by permuting dynasty labels within each couplet type and
comparing the variance of the five gaps. Writes
results/couplet_controls/dynasty_heterogeneity.json.
"""

from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

N_PERM = 2000
SEED = 0


def gaps(inner: dict, outer: dict, dynasties: list[str], sign: int) -> list[float]:
    return [sign * (st.mean(outer[d]) - st.mean(inner[d])) for d in dynasties]


def group(rows: list[dict], id2dyn: dict, value) -> dict:
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(id2dyn[r["id"]], []).append(value(r))
    return out


def permutation_p(inner: dict, outer: dict, dynasties: list[str], sign: int, rng: random.Random) -> float:
    observed = st.pvariance(gaps(inner, outer, dynasties, sign))
    flat_i = [(d, v) for d in dynasties for v in inner[d]]
    flat_o = [(d, v) for d in dynasties for v in outer[d]]
    hits = 0
    for _ in range(N_PERM):
        li = [d for d, _ in flat_i]
        lo = [d for d, _ in flat_o]
        rng.shuffle(li)
        rng.shuffle(lo)
        gi: dict[str, list[float]] = {}
        go: dict[str, list[float]] = {}
        for d, (_, v) in zip(li, flat_i):
            gi.setdefault(d, []).append(v)
        for d, (_, v) in zip(lo, flat_o):
            go.setdefault(d, []).append(v)
        hits += st.pvariance(gaps(gi, go, dynasties, sign)) >= observed
    return (hits + 1) / (N_PERM + 1)


def main() -> None:
    rng = random.Random(SEED)
    report = {"n_permutations": N_PERM, "seed": SEED, "meters": {}}
    for meter in ["penta", "hepta"]:
        test = json.loads(Path(f"data/2-3_regulated_verse/couplets_{meter}_test.json").read_text(encoding="utf-8"))
        couplet2dyn = {it["couplet"]: it["dynasty"] for it in test}
        id2dyn = {}
        for s in ["inner", "outer"]:
            for it in json.loads(Path(f"data/2-3_regulated_verse/couplet_controls/{meter}_{s}.json").read_text(encoding="utf-8")):
                id2dyn[it["id"]] = couplet2dyn[it["couplet"]]
        dec = json.loads(Path(f"results/couplet_controls/decoding_{meter}.json").read_text(encoding="utf-8"))["scores"]
        ali = json.loads(Path(f"results/couplet_controls/alignment_{meter}.json").read_text(encoding="utf-8"))["alignment"]
        measures = {
            # sign=+1: outer minus inner (entropy, loss); sign=-1: inner minus outer (alignment)
            "entropy": (group(dec["finetuned/I/inner"], id2dyn, lambda r: st.mean(r["entropy"])),
                        group(dec["finetuned/O/outer"], id2dyn, lambda r: st.mean(r["entropy"])), 1),
            "loss": (group(dec["finetuned/I/inner"], id2dyn, lambda r: st.mean(r["loss"])),
                     group(dec["finetuned/O/outer"], id2dyn, lambda r: st.mean(r["loss"])), 1),
            # alignment is stored per layer (index 0 = embedding layer)
            "alignment_layer8": (group(ali["inner"], id2dyn, lambda r: r["alignment"][8]),
                                 group(ali["outer"], id2dyn, lambda r: r["alignment"][8]), -1),
        }
        out = {}
        for name, (inner, outer, sign) in measures.items():
            dynasties = sorted(inner)
            g = gaps(inner, outer, dynasties, sign)
            out[name] = {
                "gap_by_dynasty": dict(zip(dynasties, [round(x, 4) for x in g])),
                "n_per_dynasty": {d: len(inner[d]) for d in dynasties},
                "heterogeneity_p": round(permutation_p(inner, outer, dynasties, sign, rng), 4),
            }
            print(meter, name, out[name])
        report["meters"][meter] = out
    Path("results/couplet_controls/dynasty_heterogeneity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
