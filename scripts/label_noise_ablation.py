"""Label-noise ablation for the inner/outer couplet contrast.

Follows the network filter of Kurzynski, Xu and Feng (2025, "The Game of Keys
and Queries", section 3):

1. Build a character network from the *training* inner couplets of each meter:
   an edge joins two characters that occupy the same position in the two lines
   (weighted by frequency).
2. Iteratively remove nodes of degree 1, then partition the core network with
   the Louvain method.
3. Flag an inner couplet as likely non-parallel when at least 4 of its
   position pairs do not fall in the same community, and an outer couplet as
   likely parallel when at least 3 (pentasyllabic) or 4 (heptasyllabic) of its
   pairs fall in the same community. The pentasyllabic thresholds (4 and 3 of
   5) are those of the original study; for heptasyllabic lines we keep absolute
   counts (4 and 4 of 7) rather than scaling the shares, which would give 6 and
   5 of 7 and flag almost nothing. A pair with a character outside the core
   network counts as "not in the same community".

The flags are then applied to the scored test couplets of the positional
analysis (1,000 inner and 1,000 outer per meter), and the inner/outer gaps in
entropy, loss (untuned and fine-tuned GujiGPT) and transition alignment
(GujiBERT layer 8) are recomputed on the cleaned subsets.

Run with the project venv (needs networkx):
    venv/bin/python scripts/label_noise_ablation.py
Writes results/label_noise_ablation/summary.json and summary.md.
"""

from __future__ import annotations

import json
import math
import random
import statistics as st
from collections import Counter
from pathlib import Path

import networkx as nx

SEED = 0
N_PERM = 10000
# line length -> (min. mismatched pairs to flag an inner couplet,
#                 min. matched pairs to flag an outer couplet)
THRESHOLDS = {5: (4, 3), 7: (4, 4)}
OUT = Path("results/label_noise_ablation")


def lines(couplet: str) -> tuple[str, str]:
    a, b = couplet.split("，")
    return a, b


def build_communities(train: list[dict]) -> tuple[dict[str, int], dict]:
    g = nx.Graph()
    for it in train:
        if it["label"] != "i":
            continue
        a, b = lines(it["couplet"])
        for x, y in zip(a, b):
            if x == y:
                continue
            w = g.get_edge_data(x, y, {"weight": 0})["weight"]
            g.add_edge(x, y, weight=w + 1)
    full_nodes = g.number_of_nodes()
    while True:
        leaves = [n for n, d in g.degree() if d <= 1]
        if not leaves:
            break
        g.remove_nodes_from(leaves)
    comms = nx.community.louvain_communities(g, weight="weight", seed=SEED)
    node2comm = {n: i for i, c in enumerate(comms) for n in c}
    info = {"nodes_full": full_nodes, "nodes_core": g.number_of_nodes(),
            "communities": len(comms), "community_sizes": sorted((len(c) for c in comms), reverse=True)[:12]}
    return node2comm, info


def same_count(couplet: str, node2comm: dict[str, int]) -> tuple[int, int]:
    a, b = lines(couplet)
    same = sum(1 for x, y in zip(a, b) if x in node2comm and y in node2comm and node2comm[x] == node2comm[y])
    return same, len(a)


def flagged(item: dict, label: str, node2comm: dict[str, int]) -> bool:
    same, n = same_count(item["couplet"], node2comm)
    inner_min_mismatch, outer_min_match = THRESHOLDS[n]
    if label == "i":
        return n - same >= inner_min_mismatch
    return same >= outer_min_match


def perm_p(x: list[float], y: list[float], rng: random.Random) -> float:
    obs = abs(st.mean(x) - st.mean(y))
    pool = x + y
    n = len(x)
    hits = 0
    for _ in range(N_PERM):
        rng.shuffle(pool)
        hits += abs(st.mean(pool[:n]) - st.mean(pool[n:])) >= obs
    return (hits + 1) / (N_PERM + 1)


def main() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"seed": SEED, "thresholds": {str(k): {"inner_min_mismatched_pairs": v[0], "outer_min_matched_pairs": v[1]}
                                         for k, v in THRESHOLDS.items()}, "meters": {}}
    md = ["# Label-noise ablation (network filter after Kurzynski et al. 2025)", ""]
    for meter in ["penta", "hepta"]:
        train = json.loads(Path(f"data/2-3_regulated_verse/couplets_{meter}_train.json").read_text(encoding="utf-8"))
        test = json.loads(Path(f"data/2-3_regulated_verse/couplets_{meter}_test.json").read_text(encoding="utf-8"))
        node2comm, info = build_communities(train)

        def rate(items: list[dict], lab: str) -> float:
            sel = [it for it in items if it["label"] == lab]
            return sum(flagged(it, lab, node2comm) for it in sel) / len(sel)

        rates = {"train_inner_flagged": rate(train, "i"), "train_outer_flagged": rate(train, "o"),
                 "test_inner_flagged": rate(test, "i"), "test_outer_flagged": rate(test, "o")}

        sets = {s: json.loads(Path(f"data/2-3_regulated_verse/couplet_controls/{meter}_{s}.json").read_text(encoding="utf-8"))
                for s in ["inner", "outer"]}
        keep = {s: {it["id"] for it in sets[s] if not flagged(it, "i" if s == "inner" else "o", node2comm)}
                for s in sets}
        dec = json.loads(Path(f"results/couplet_controls/decoding_{meter}.json").read_text(encoding="utf-8"))["scores"]
        ali = json.loads(Path(f"results/couplet_controls/alignment_{meter}.json").read_text(encoding="utf-8"))["alignment"]

        measures = {
            # (inner rows, outer rows, value, sign): gap = sign * (outer - inner), positive = inner more constrained
            "entropy_finetuned": (dec["finetuned/I/inner"], dec["finetuned/O/outer"], lambda r: st.mean(r["entropy"]), 1),
            "loss_finetuned": (dec["finetuned/I/inner"], dec["finetuned/O/outer"], lambda r: st.mean(r["loss"]), 1),
            "entropy_untuned": (dec["base/none/inner"], dec["base/none/outer"], lambda r: st.mean(r["entropy"]), 1),
            "loss_untuned": (dec["base/none/inner"], dec["base/none/outer"], lambda r: st.mean(r["loss"]), 1),
            "alignment_layer8": (ali["inner"], ali["outer"], lambda r: r["alignment"][8], -1),
        }
        res = {}
        for name, (irows, orows, f, sign) in measures.items():
            out = {}
            for cond, ki, ko in [("all", None, None), ("cleaned", keep["inner"], keep["outer"])]:
                xi = [f(r) for r in irows if ki is None or r["id"] in ki]
                xo = [f(r) for r in orows if ko is None or r["id"] in ko]
                out[cond] = {"n_inner": len(xi), "n_outer": len(xo),
                             "gap": round(sign * (st.mean(xo) - st.mean(xi)), 4),
                             "p": round(perm_p(xi, xo, rng), 4)}
            res[name] = out
        report["meters"][meter] = {"network": info, "flag_rates": {k: round(v, 4) for k, v in rates.items()},
                                   "kept": {s: len(v) for s, v in keep.items()}, "gaps": res}

        md += [f"## {meter}", "",
               f"Network: {info['nodes_full']} characters, core {info['nodes_core']}, {info['communities']} communities.",
               f"Flagged as mislabeled: train inner {rates['train_inner_flagged']:.1%}, train outer {rates['train_outer_flagged']:.1%}; "
               f"test inner {rates['test_inner_flagged']:.1%}, test outer {rates['test_outer_flagged']:.1%}.",
               f"Scored sets kept: inner {len(keep['inner'])}/1000, outer {len(keep['outer'])}/1000.", "",
               "| Measure | Gap, all | Gap, cleaned | p, cleaned |", "|:--|--:|--:|--:|"]
        for name, out in res.items():
            md.append(f"| {name} | {out['all']['gap']:+.3f} | {out['cleaned']['gap']:+.3f} | {out['cleaned']['p']:.4f} |")
        md += ["", "Gap: inner more constrained is positive (outer minus inner for entropy and loss; inner minus outer for alignment).", ""]

    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
