"""Nonsense counterparts of the 欽定四書文 essays with hand-marked leg pairs.

Two methods, chosen with --method:

- fragments (default): every pair of corresponding fragments (fragment k of
  leg A and fragment k of leg B of a pair) is replaced by a pair of
  corresponding fragments taken whole from a leg pair of another essay of the
  corpus, a different essay for every pair of fragments of the essay, so that
  every vertical pair is a real parallel pair while no leg makes sense: no two
  fragments of a leg, or of the whole essay, come from the same essay. This is
  the essay counterpart of the segment-swapped couplets
  (make_couplet_controls.py segswap). Donor fragments have the same lengths as
  the fragments they replace wherever an essay not yet used in this
  counterpart offers such a pair (the slots with the fewest such donors are
  filled first); otherwise the closest lengths are taken. A leg never repeats
  a fragment immediately, since a zero transition has no direction. The
  fragments of a leg are joined with "，"; the opening sections and the
  conclusion are not built, and "text" holds the eight legs only.
- characters: every essay is filled with random unrelated characters as in
  make_nonsense_controls.py: each distinct character of the essay is replaced
  by a different character drawn at random from the --max-rank most frequent
  characters of the full Mengzi, excluding the characters of that essay and
  any character outside the GujiBERT vocabulary. The same mapping is applied
  to the essay's leg pairs, so that repeated characters stay repeated within
  and across the two members of a pair and punctuation, and with it the
  fragments, is kept.
- shuffle: in every pair, the fragments of leg B are permuted (a derangement
  wherever the fragments allow one, never the leg itself, and never a fragment
  repeated immediately) while leg A is kept, so that both legs keep their
  wording but leg B no longer answers leg A fragment by fragment. This is the
  essay counterpart of the char-shuffled couplets (make_couplet_controls.py
  charshuf). The fragments of leg B are joined with "，".

Essay i uses seed --seed + i. Writes --output in the format of
data/4_eight_legged_essays/qinding_legs.json, so that baguwen_pseudo_legs.py and
baguwen_leg_streams.py can read it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from baguwen_fragment_space import FRAGMENT_SPLIT


def is_han(char: str) -> bool:
    return "一" <= char <= "鿿" or "㐀" <= char <= "䶿"


def split(text: str) -> list[str]:
    return [f for f in FRAGMENT_SPLIT.split(text) if f]


def character_nonsense(essays: list[dict], args) -> list[dict]:
    """Each distinct character of an essay replaced by a random frequent character of the Mengzi."""
    from transformers import AutoTokenizer

    from make_nonsense_controls import build_pool
    from self_similarity_matrix import DEFAULT_MODEL

    vocab = AutoTokenizer.from_pretrained(args.model or DEFAULT_MODEL).get_vocab()
    output = []
    for index, essay in enumerate(essays):
        sources = essay["text"] + "".join(p["a"] + p["b"] for p in essay["pairs"])
        chars = list(dict.fromkeys(c for c in sources if is_han(c)))
        pool = build_pool(Path(args.pool_json), args.max_rank, set(chars), vocab)
        seed = args.seed + index
        mapping = dict(zip(chars, random.Random(seed).sample(pool, len(chars))))
        substitute = lambda text: "".join(mapping.get(c, c) for c in text)
        output.append({"id": essay["id"], "template": essay["id"], "seed": seed,
                       "text": substitute(essay["text"]),
                       "pairs": [{"a": substitute(p["a"]), "b": substitute(p["b"]), "fragments": p["fragments"]}
                                 for p in essay["pairs"]]})
    return output


def shuffle_fragments(fragments: list[str], rng: random.Random, tries: int = 500) -> list[str]:
    """A permutation of the fragments with as few left in place as possible, never the original order,
    and without a fragment repeated immediately."""
    best = None
    for _ in range(tries):
        order = list(range(len(fragments)))
        rng.shuffle(order)
        new = [fragments[i] for i in order]
        if new == fragments or any(x == y for x, y in zip(new[:-1], new[1:])):
            continue
        fixed = sum(x == y for x, y in zip(new, fragments))
        if best is None or fixed < best[0]:
            best = (fixed, new)
            if fixed == 0:
                break
    if best is None:
        raise ValueError(f"No admissible permutation of {fragments}")
    return best[1]


def shuffled_legs(essays: list[dict], args) -> list[dict]:
    """Leg B of every pair with its fragments permuted, leg A unchanged."""
    output = []
    for index, essay in enumerate(essays):
        seed = args.seed + index
        rng = random.Random(seed)
        pairs = []
        for pair in essay["pairs"]:
            a, b = split(pair["a"]), split(pair["b"])
            if len(b) < 2:
                raise ValueError(f"Essay {essay['id']}: a leg with fewer than two fragments cannot be shuffled")
            pairs.append({"a": pair["a"], "b": "，".join(shuffle_fragments(b, rng)), "fragments": pair["fragments"]})
        output.append({"id": essay["id"], "template": essay["id"], "seed": seed,
                       "text": "。".join(pair[key] for pair in pairs for key in ("a", "b")) + "。", "pairs": pairs})
    return output


def fragment_nonsense(essays: list[dict], args) -> tuple[list[dict], dict]:
    """Corresponding fragment pairs of every essay replaced by those of other essays, one donor essay per pair."""
    donors = [(essay["id"], a, b) for essay in essays for pair in essay["pairs"]
              for a, b in zip(split(pair["a"]), split(pair["b"]))]
    by_lengths: dict[tuple[int, int], list[tuple[str, str, str]]] = {}
    for donor in donors:
        by_lengths.setdefault((len(donor[1]), len(donor[2])), []).append(donor)
    output, exact, total, attempts_needed = [], 0, 0, []
    for index, essay in enumerate(essays):
        seed = args.seed + index
        rng = random.Random(seed)
        legs = [(split(pair["a"]), split(pair["b"])) for pair in essay["pairs"]]
        for a, b in legs:
            if len(a) != len(b):
                raise ValueError(f"Essay {essay['id']}: legs with {len(a)} and {len(b)} fragments")
        slots = [(p, k, len(a[k]), len(b[k])) for p, (a, b) in enumerate(legs) for k in range(len(a))]
        for attempt in range(1, args.max_attempts + 1):
            # The slots with the fewest exact-length donors are filled first, ties in random order.
            order = sorted(slots, key=lambda s: (len({d[0] for d in by_lengths.get(s[2:], []) if d[0] != essay["id"]}),
                                                 rng.random()))
            used, chosen, matched = {essay["id"]}, {}, 0
            for slot in order:
                candidates = [d for d in by_lengths.get(slot[2:], []) if d[0] not in used]
                if candidates:
                    matched += 1
                else:
                    gap = lambda d: abs(len(d[1]) - slot[2]) + abs(len(d[2]) - slot[3])
                    free = [d for d in donors if d[0] not in used]
                    closest = min(map(gap, free))
                    candidates = [d for d in free if gap(d) == closest]
                donor = rng.choice(candidates)
                chosen[slot[:2]] = donor
                used.add(donor[0])
            pairs = []
            for p, (a, _) in enumerate(legs):
                picked = [chosen[(p, k)] for k in range(len(a))]
                pairs.append({"a": "，".join(d[1] for d in picked), "b": "，".join(d[2] for d in picked),
                              "fragments": len(picked), "donors": [d[0] for d in picked]})
            repeats = any(x == y for pair in pairs for key in ("a", "b")
                          for x, y in zip(split(pair[key])[:-1], split(pair[key])[1:]))
            if not repeats:
                break
        else:
            raise RuntimeError(f"Essay {essay['id']}: a leg repeats a fragment after {args.max_attempts} attempts")
        attempts_needed.append(attempt)
        exact += matched
        total += len(slots)
        output.append({"id": essay["id"], "template": essay["id"], "seed": seed,
                       "text": "。".join(pair[key] for pair in pairs for key in ("a", "b")) + "。", "pairs": pairs})
    stats = {"fragment_pairs": total, "exact_length_matches": exact,
             "essays_needing_redraw": sum(a > 1 for a in attempts_needed)}
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/4_eight_legged_essays/qinding_legs.json")
    parser.add_argument("--method", choices=["fragments", "characters", "shuffle"], default="fragments")
    parser.add_argument("--pool-json", default="data/1_mengzi_prose/mengzi_full.json", help="characters method only")
    parser.add_argument("--max-rank", type=int, default=1000, help="characters method only")
    parser.add_argument("--model", default=None, help="characters method only (default: GujiBERT)")
    parser.add_argument("--max-attempts", type=int, default=50, help="fragments method: redraws per essay")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="data/4_eight_legged_essays/qinding_controls/nonsense.json")
    args = parser.parse_args()

    essays = json.loads(Path(args.corpus).read_text(encoding="utf-8"))["essays"]
    payload = {"generator": "make_qinding_nonsense.py", "method": args.method, "template": args.corpus}
    if args.method == "fragments":
        output_essays, stats = fragment_nonsense(essays, args)
        payload["note"] = ("Corresponding fragment pairs taken whole from leg pairs of other essays, a different "
                           "essay for every pair of fragments; donors in 'donors', per fragment")
        payload["stats"] = stats
        print(f"{stats['exact_length_matches']}/{stats['fragment_pairs']} fragment pairs with the lengths of the "
              f"pair they replace; {stats['essays_needing_redraw']} essays redrawn to avoid repeated fragments")
    elif args.method == "shuffle":
        output_essays = shuffled_legs(essays, args)
        payload["note"] = "Fragments of leg B permuted in every pair; leg A unchanged"
    else:
        output_essays = character_nonsense(essays, args)
        payload.update({"pool_json": args.pool_json, "max_rank": args.max_rank})
    payload["essays"] = output_essays

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(output_essays)} essays, {sum(len(e['pairs']) for e in output_essays)} pairs: {output}")


if __name__ == "__main__":
    main()
