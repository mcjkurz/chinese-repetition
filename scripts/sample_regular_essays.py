"""Sample the regular essays of 欽定四書文 used in the analysis, as evenly across periods as the data allow.

Reads data/4_eight_legged_essays/qinding_regular_legs.json (prepare_qinding_legs.py on the regular
annotations) and keeps --size essays: every regular essay of a period with
fewer than its even share, the remaining places divided evenly among the
other periods and filled at random (--seed). Essays given with --include
(by default Chen Zilong's, no. 346) are always kept and count towards their
period. A sampled essay with a leg of fewer than --min-fragments clauses has
no clause transition in that pair and is replaced by an eligible essay of the
same period drawn at random (seed --seed + 1), so that the sample of the
first draw is otherwise kept. Writes the sample in the same format.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PERIODS = ["化治", "正嘉", "隆萬", "啟禎", "本朝"]


def quotas(available: dict[str, int], size: int) -> dict[str, int]:
    """Even shares, with periods that cannot fill theirs giving all they have and the rest shared out."""
    quota, open_periods, left = {}, [p for p in PERIODS if available.get(p)], size
    while open_periods:
        share = left / len(open_periods)
        short = [p for p in open_periods if available[p] <= share]
        if not short:
            base, extra = divmod(left, len(open_periods))
            # Leftover places go to the periods with most essays.
            for rank, p in enumerate(sorted(open_periods, key=lambda p: -available[p])):
                quota[p] = base + (rank < extra)
            break
        for p in short:
            quota[p] = available[p]
            left -= available[p]
            open_periods.remove(p)
    return quota


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/4_eight_legged_essays/qinding_regular_legs.json"))
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--include", nargs="*", default=["346"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-fragments", type=int, default=2,
                        help="Clauses every leg must have (0 to keep the first draw as it is)")
    parser.add_argument("--output", type=Path, default=Path("data/4_eight_legged_essays/qinding_regular100/legs.json"))
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    by_period: dict[str, list[dict]] = {}
    for essay in corpus["essays"]:
        by_period.setdefault(essay["date"], []).append(essay)
    quota = quotas({p: len(v) for p, v in by_period.items()}, args.size)
    rng = random.Random(args.seed)
    sample = []
    for period in PERIODS:
        essays = by_period.get(period, [])
        forced = [e for e in essays if e["id"] in args.include]
        rest = [e for e in essays if e["id"] not in args.include]
        sample += forced + rng.sample(rest, quota.get(period, 0) - len(forced))
        print(f"{period}: {quota.get(period, 0)} of {len(essays)}")
    eligible = lambda e: all(p["fragments"] >= args.min_fragments for p in e["pairs"])
    replace_rng, chosen = random.Random(args.seed + 1), {e["id"] for e in sample}
    for index, essay in enumerate(sample):
        if eligible(essay):
            continue
        pool = [e for e in by_period[essay["date"]] if e["id"] not in chosen and eligible(e)]
        sample[index] = replace_rng.choice(pool)
        chosen.add(sample[index]["id"])
        print(f"{essay['date']}: essay {essay['id']} has a leg of fewer than {args.min_fragments} clauses, "
              f"replaced by {sample[index]['id']}")
    sample.sort(key=lambda e: int(e["id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source": corpus["source"], "sample": f"{args.size} regular essays, as evenly across periods as "
               f"possible (seed {args.seed}, always including {', '.join(args.include)}; every leg "
               f"at least {args.min_fragments} clauses)", "essays": sample}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(sample)} essays, {sum(len(e['pairs']) for e in sample)} pairs: {args.output}")


if __name__ == "__main__":
    main()
