"""Prepare couplet alignment data from the regulated-poem CSV.

By default this creates data/2-3_regulated_verse/couplets_penta_alignment.json and
data/2-3_regulated_verse/couplets_hepta_alignment.json, each with 100,000 line-disjoint couplets
sampled equally across the five target dynasties and positions 1-4. Use
--dynasty to filter to one dynasty, and --positions to keep only selected
couplet positions.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import opencc
from tqdm import tqdm

from common import METER_NAMES, METERS, parse_regulated_couplets, poem_id_for, split_couplet


DEFAULT_INPUT_PATH = "data/2-3_regulated_verse/regulated.csv"
DEFAULT_OUTPUT_TEMPLATE = "data/2-3_regulated_verse/couplets_{meter}_alignment.json"
DEFAULT_MAX_SAMPLES = 100_000
DEFAULT_POSITIONS = (1, 2, 3, 4)
DYNASTIES = ("唐", "宋", "元", "明", "清")
DEFAULT_SAMPLING_MODE = "position-dynasty-balanced"
SAMPLING_MODES = (
    DEFAULT_SAMPLING_MODE,
    "position-balanced",
    "dynasty-balanced",
    "random",
)


def make_record(
    couplet: str,
    position: int,
    dynasty: str,
    meter: int,
    poem_id: str,
    converter: opencc.OpenCC,
) -> dict | None:
    """Return one alignment record with simplified and traditional fields."""
    line1, line2 = split_couplet(couplet)
    if line1 is None or line2 is None:
        return None

    line1_traditional = converter.convert(line1)
    line2_traditional = converter.convert(line2)
    return {
        "couplet": couplet,
        "couplet_traditional": f"{line1_traditional}，{line2_traditional}",
        "line1": line1,
        "line2": line2,
        "line1_traditional": line1_traditional,
        "line2_traditional": line2_traditional,
        "position": position,
        "dynasty": dynasty,
        "meter": meter,
        "poem_id": poem_id,
    }


def sampling_key(item: dict, sampling_mode: str) -> tuple | None:
    """Return the balance bucket key for a sampled record."""
    if sampling_mode == "position-dynasty-balanced":
        return (item["dynasty"], item["position"])
    if sampling_mode == "position-balanced":
        return (item["position"],)
    if sampling_mode == "dynasty-balanced":
        return (item["dynasty"],)
    if sampling_mode == "random":
        return None
    raise ValueError(f"Unsupported sampling mode: {sampling_mode}")


def expected_bucket_keys(
    sampling_mode: str,
    positions: tuple[int, ...],
    dynasty: str | None,
) -> list[tuple]:
    """Return requested balance buckets for the active dynasty/position filters."""
    dynasties = (dynasty,) if dynasty is not None else DYNASTIES
    if sampling_mode == "position-dynasty-balanced":
        return [(item_dynasty, position) for item_dynasty in dynasties for position in positions]
    if sampling_mode == "position-balanced":
        return [(position,) for position in positions]
    if sampling_mode == "dynasty-balanced":
        return [(item_dynasty,) for item_dynasty in dynasties]
    if sampling_mode == "random":
        return []
    raise ValueError(f"Unsupported sampling mode: {sampling_mode}")


def target_counts(max_samples: int, keys: list[tuple], rng: random.Random) -> dict[tuple, int]:
    """Return near-equal bucket targets, distributing any remainder deterministically."""
    base = max_samples // len(keys)
    remainder = max_samples % len(keys)
    shuffled_keys = keys.copy()
    rng.shuffle(shuffled_keys)
    targets = {key: base for key in keys}
    for key in shuffled_keys[:remainder]:
        targets[key] += 1
    return targets


def sample_records(
    records: list[dict],
    max_samples: int | None,
    sampling_mode: str,
    positions: tuple[int, ...],
    dynasty: str | None,
    seed: int,
) -> list[dict]:
    """Sample records according to the requested balance mode, filling shortages."""
    rng = random.Random(seed)
    records = records.copy()
    if max_samples is None:
        rng.shuffle(records)
        return records
    if max_samples >= len(records):
        rng.shuffle(records)
        if len(records) < max_samples:
            print(f"Only {len(records)} valid records available; requested {max_samples}")
        return records

    if sampling_mode == "random":
        rng.shuffle(records)
        return records[:max_samples]

    keys = expected_bucket_keys(sampling_mode, positions, dynasty)
    targets = target_counts(max_samples, keys, rng)
    buckets = {key: [] for key in keys}
    for record in records:
        key = sampling_key(record, sampling_mode)
        if key in buckets:
            buckets[key].append(record)

    selected = []
    selected_ids = set()
    shortages = {}
    for key in keys:
        items = buckets[key]
        rng.shuffle(items)
        target = targets[key]
        chosen = items[:target]
        selected.extend(chosen)
        selected_ids.update(id(item) for item in chosen)
        if len(chosen) < target:
            shortages[key] = target - len(chosen)

    if shortages:
        print(f"Could not fill balanced buckets exactly; shortages: {shortages}")

    if len(selected) < max_samples:
        remainder = [record for record in records if id(record) not in selected_ids]
        rng.shuffle(remainder)
        selected.extend(remainder[: max_samples - len(selected)])

    rng.shuffle(selected)
    return selected[:max_samples]


def collect_alignment_records(
    input_path: Path,
    meter: int,
    dynasty: str | None,
    positions: tuple[int, ...],
    max_samples: int | None,
    seed: int,
    line_disjoint: bool,
    sampling_mode: str,
) -> tuple[list[dict], Counter]:
    """Collect alignment records from CSV rows matching the requested filters."""
    converter = opencc.OpenCC("s2t")
    collected = []
    seen_couplets = set()
    seen_lines = set()
    rng = random.Random(seed)
    stats = Counter()

    with input_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rng.shuffle(rows)

    for row in tqdm(rows, desc="Processing poems"):
        if int(row["meter"]) != meter:
            continue
        row_dynasty = row["dynasty"]
        if dynasty is not None and row_dynasty != dynasty:
            continue
        if dynasty is None and row_dynasty not in DYNASTIES:
            stats["unsupported_dynasty"] += 1
            continue

        stats["matching_poems"] += 1
        poem_id = poem_id_for(meter, row["content"])
        for couplet, position in parse_regulated_couplets(row["content"], meter):
            if position not in positions:
                continue
            if couplet in seen_couplets:
                stats["duplicate_couplets"] += 1
                continue

            record = make_record(couplet, position, row_dynasty, meter, poem_id, converter)
            if record is None:
                stats["malformed_couplets"] += 1
                continue
            if line_disjoint and (
                record["line1"] in seen_lines or record["line2"] in seen_lines
            ):
                stats["duplicate_lines"] += 1
                continue

            seen_couplets.add(couplet)
            seen_lines.update((record["line1"], record["line2"]))
            collected.append(record)

    records = sample_records(collected, max_samples, sampling_mode, positions, dynasty, seed)
    return records, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare regulated-verse couplet alignment data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(DEFAULT_INPUT_PATH),
        help="Input regulated.csv path",
    )
    parser.add_argument(
        "--meters",
        type=int,
        nargs="+",
        choices=METERS,
        default=list(METERS),
        help="Which meters to prepare alignment data for (default: both 5 and 7)",
    )
    parser.add_argument(
        "--output-path-template",
        type=str,
        default=DEFAULT_OUTPUT_TEMPLATE,
        help=(
            "Template for per-meter output JSON paths; {meter} is replaced "
            f"with 'penta' or 'hepta' (default: {DEFAULT_OUTPUT_TEMPLATE})"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Total records to sample equally across positions, per meter",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Write all matching records instead of sampling --max-samples",
    )
    parser.add_argument(
        "--dynasty",
        default=None,
        help="Optional dynasty value to keep, for example 唐",
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=list(DEFAULT_POSITIONS),
        help="Couplet positions to keep",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for row sampling and output order",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=SAMPLING_MODES,
        default=DEFAULT_SAMPLING_MODE,
        help=(
            "Sampling strategy. Default balances across dynasty and position "
            "buckets, then fills shortages from any valid bucket."
        ),
    )
    parser.add_argument(
        "--allow-reused-lines",
        action="store_true",
        help="Allow records that share an individual line with another record",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positions = tuple(args.positions)
    if not positions:
        raise ValueError("--positions must include at least one position")

    max_samples = None if args.all else args.max_samples

    for meter in args.meters:
        print(f"\n{'=' * 72}\nMeter: {meter}言律诗\n{'=' * 72}")
        output_path = Path(args.output_path_template.format(meter=METER_NAMES[meter]))

        records, stats = collect_alignment_records(
            input_path=args.input,
            meter=meter,
            dynasty=args.dynasty,
            positions=positions,
            max_samples=max_samples,
            seed=args.seed,
            line_disjoint=not args.allow_reused_lines,
            sampling_mode=args.sampling_mode,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        position_counts = Counter(record["position"] for record in records)
        dynasty_label = args.dynasty if args.dynasty is not None else "all dynasties"
        print(f"Wrote {len(records)} records to {output_path}")
        print(f"Dynasty filter: {dynasty_label}")
        print(f"Sampling mode: {args.sampling_mode if max_samples is not None else 'all'}")
        print(f"Matching poems: {stats['matching_poems']}")
        dynasty_counts = Counter(record["dynasty"] for record in records)
        for item_dynasty in ((args.dynasty,) if args.dynasty is not None else DYNASTIES):
            print(f"  dynasty {item_dynasty}: {dynasty_counts[item_dynasty]}")
        for position in positions:
            print(f"  position {position}: {position_counts[position]}")
        print(f"Skipped duplicate couplets: {stats['duplicate_couplets']}")
        if not args.allow_reused_lines:
            print(f"Skipped duplicate lines: {stats['duplicate_lines']}")
        print(f"Skipped unsupported dynasties: {stats['unsupported_dynasty']}")
        print(f"Skipped malformed couplets: {stats['malformed_couplets']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
