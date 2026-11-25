"""
Prepare train/test corpora of Inner/Outer regulated-verse couplets.

In a 律詩 (regulated poem), each poem has 4 couplets:
- Couplet 1 (首聯): outer -> "o"
- Couplet 2 (頷聯): inner -> "i"
- Couplet 3 (頸聯): inner -> "i"
- Couplet 4 (尾聯): outer -> "o"

The split is at the poem level, for both pentasyllabic and heptasyllabic
verse. Every couplet of a poem is assigned to train or to test, never to
both. A test poem also shares no couplet string and no individual line with
any training poem.

Each corpus is balanced across dynasty. Because every selected poem contributes
all four positions, position and inner/outer counts match by construction.
Output is simplified Chinese. The finetune step converts to traditional
Chinese on the fly.
"""

import argparse
import json
import csv
import random
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from common import (
    METER_NAMES,
    io_label_for_position,
    parse_regulated_couplets,
    poem_id_for,
    split_couplet,
)


DYNASTIES = ("唐", "宋", "元", "明", "清")
POSITIONS = (1, 2, 3, 4)
METERS = (5, 7)
DEFAULT_TRAIN_SAMPLES = 100_000
DEFAULT_TEST_SAMPLES = 10_000
DEFAULT_INPUT = "data/2-3_regulated_verse/regulated.csv"
DEFAULT_OUTPUT_DIR = "data/2-3_regulated_verse"
DEFAULT_REPORT_PATH = "data/2-3_regulated_verse/split_report.json"


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


def load_poems(csv_path: str, meter: int) -> tuple[list[dict], Counter]:
    """Load complete regulated poems for one meter.

    A poem is kept only when the first four couplet slots are all present,
    so position labels 1-4 refer to 首聯/頷聯/頸聯/尾聯 of the same poem.
    """
    poems = []
    seen_ids = set()
    stats = Counter()

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in tqdm(rows, desc=f"Reading {meter}言 poems"):
        if int(row["meter"]) != meter:
            continue
        dynasty = row["dynasty"]
        if dynasty not in DYNASTIES:
            stats["unsupported_dynasty"] += 1
            continue

        content = row["content"]
        poem_id = poem_id_for(meter, content)
        if poem_id in seen_ids:
            stats["duplicate_poems"] += 1
            continue
        seen_ids.add(poem_id)

        parsed = parse_regulated_couplets(content, meter)
        if len(parsed) != 4 or [position for _, position in parsed] != list(POSITIONS):
            stats["incomplete_poems"] += 1
            continue

        couplets = []
        lines = []
        for couplet_text, position in parsed:
            line1, line2 = split_couplet(couplet_text)
            if line1 is None or line2 is None:
                couplets = []
                break
            couplets.append(
                {
                    "couplet": couplet_text,
                    "label": io_label_for_position(position),
                    "position": position,
                    "dynasty": dynasty,
                    "meter": meter,
                    "poem_id": poem_id,
                    "title": (row.get("title") or "").strip(),
                    "author": (row.get("author") or "").strip(),
                    "lines": (line1, line2),
                }
            )
            lines.extend((line1, line2))
        if len(couplets) != 4:
            stats["incomplete_poems"] += 1
            continue

        poems.append(
            {
                "poem_id": poem_id,
                "dynasty": dynasty,
                "title": (row.get("title") or "").strip(),
                "author": (row.get("author") or "").strip(),
                "couplets": couplets,
                "couplet_texts": [item["couplet"] for item in couplets],
                "lines": lines,
            }
        )
        stats["complete_poems"] += 1

    return poems, stats


def poems_by_dynasty(poems: list[dict]) -> dict[str, list[dict]]:
    """Group poems by dynasty, preserving input order within each dynasty."""
    grouped = {dynasty: [] for dynasty in DYNASTIES}
    for poem in poems:
        grouped[poem["dynasty"]].append(poem)
    return grouped


def select_poems(
    grouped: dict[str, list[dict]],
    n_poems: int,
    seed: int,
    blocked_ids: set[str],
    blocked_couplets: set[str],
    blocked_lines: set[str],
) -> tuple[list[dict], dict, int]:
    """Take whole poems, balanced by dynasty, skipping any blocked overlap.

    `blocked_*` are updated in place so a later call (the other split) cannot
    reuse a poem, a couplet string, or a line.
    """
    rng = random.Random(seed)
    pools = {}
    for dynasty in DYNASTIES:
        items = grouped[dynasty].copy()
        rng.shuffle(items)
        pools[dynasty] = items

    targets = target_counts(n_poems, [(dynasty,) for dynasty in DYNASTIES], rng)
    selected = []
    shortages = {}
    rejected_overlap = 0

    def accept(poem: dict) -> bool:
        nonlocal rejected_overlap
        if poem["poem_id"] in blocked_ids:
            return False
        overlaps = (
            any(text in blocked_couplets for text in poem["couplet_texts"])
            or any(line in blocked_lines for line in poem["lines"])
        )
        if overlaps:
            rejected_overlap += 1
            return False
        selected.append(poem)
        blocked_ids.add(poem["poem_id"])
        blocked_couplets.update(poem["couplet_texts"])
        blocked_lines.update(poem["lines"])
        return True

    for dynasty in DYNASTIES:
        need = targets[(dynasty,)]
        got = 0
        for poem in pools[dynasty]:
            if got >= need:
                break
            if accept(poem):
                got += 1
        if got < need:
            shortages[dynasty] = need - got

    if len(selected) < n_poems:
        leftover = [poem for dynasty in DYNASTIES for poem in pools[dynasty]]
        rng.shuffle(leftover)
        for poem in leftover:
            if len(selected) >= n_poems:
                break
            accept(poem)

    return selected, shortages, rejected_overlap


def couplet_records(poems: list[dict], seed: int) -> list[dict]:
    """Flatten selected poems into couplet records, in a shuffled order."""
    records = []
    for poem in poems:
        for couplet in poem["couplets"]:
            records.append({key: value for key, value in couplet.items() if key != "lines"})
    rng = random.Random(seed)
    rng.shuffle(records)
    return records


def overlap_counts(train_poems: list[dict], test_poems: list[dict]) -> dict:
    """Count poem, couplet, and line overlap between the two splits."""
    train_ids = {poem["poem_id"] for poem in train_poems}
    test_ids = {poem["poem_id"] for poem in test_poems}
    train_couplets = {text for poem in train_poems for text in poem["couplet_texts"]}
    test_couplets = {text for poem in test_poems for text in poem["couplet_texts"]}
    train_lines = {line for poem in train_poems for line in poem["lines"]}
    test_lines = {line for poem in test_poems for line in poem["lines"]}
    return {
        "shared_poem_ids": len(train_ids & test_ids),
        "shared_couplets": len(train_couplets & test_couplets),
        "shared_lines": len(train_lines & test_lines),
    }


def dynasty_counts(poems: list[dict]) -> dict[str, int]:
    """Return poem counts for the five dynasties."""
    counts = Counter(poem["dynasty"] for poem in poems)
    return {dynasty: counts[dynasty] for dynasty in DYNASTIES}


def build_output_path(output_dir: str, meter: int, is_test: bool) -> str:
    """Return the default corpus path for a meter and split."""
    split = "test" if is_test else "train"
    return f"{output_dir}/couplets_{METER_NAMES[meter]}_{split}.json"


def poem_target(couplet_target: int) -> int:
    """Return how many whole poems are needed for a couplet budget."""
    if couplet_target % 4 != 0:
        raise ValueError(
            f"Sample count must be a multiple of 4 so every poem stays intact "
            f"(got {couplet_target})."
        )
    return couplet_target // 4


def main():
    parser = argparse.ArgumentParser(
        description="Prepare poem-level Inner/Outer couplet corpora"
    )
    parser.add_argument(
        "--meters",
        type=int,
        nargs="+",
        choices=METERS,
        default=list(METERS),
        help="Which meters to build corpora for (default: both 5 and 7)",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES,
        help=f"Training couplets per meter (default: {DEFAULT_TRAIN_SAMPLES})",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=DEFAULT_TEST_SAMPLES,
        help=f"Test couplets per meter (default: {DEFAULT_TEST_SAMPLES})",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Input regulated-verse CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the output train/test JSON files (default: data/2-3_regulated_verse)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help=f"Where to write the split report (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=42,
        help="Random seed for the training poems (default: 42)",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=123,
        help="Random seed for the held-out poems (default: 123)",
    )
    args = parser.parse_args()

    train_poems_n = poem_target(args.train_samples)
    test_poems_n = poem_target(args.test_samples)
    report = {
        "split_unit": "poem",
        "definition": (
            "A poem is the meter plus the normalized full text. Every couplet "
            "of a poem is assigned to the same split. Test poems share no poem "
            "id, couplet string, or line with the training poems."
        ),
        "train_couplets_requested": args.train_samples,
        "test_couplets_requested": args.test_samples,
        "meters": {},
    }

    for meter in args.meters:
        print(f"\n{'=' * 72}\nMeter: {meter}言律诗\n{'=' * 72}")
        poems, stats = load_poems(args.input, meter)
        print(
            f"Complete poems: {stats['complete_poems']:,} "
            f"(dropped {stats['incomplete_poems']:,} incomplete, "
            f"{stats['duplicate_poems']:,} duplicate texts)"
        )
        grouped = poems_by_dynasty(poems)
        for dynasty in DYNASTIES:
            print(f"  available {dynasty}: {len(grouped[dynasty]):,}")

        blocked_ids: set[str] = set()
        blocked_couplets: set[str] = set()
        blocked_lines: set[str] = set()
        test_poems, test_shortages, test_rejected = select_poems(
            grouped,
            test_poems_n,
            args.test_seed,
            blocked_ids,
            blocked_couplets,
            blocked_lines,
        )
        train_poems, train_shortages, train_rejected = select_poems(
            grouped,
            train_poems_n,
            args.train_seed,
            blocked_ids,
            blocked_couplets,
            blocked_lines,
        )
        overlaps = overlap_counts(train_poems, test_poems)
        if any(overlaps.values()):
            raise RuntimeError(f"Poem-level split failed an overlap check: {overlaps}")

        train_records = couplet_records(train_poems, args.train_seed)
        test_records = couplet_records(test_poems, args.test_seed)
        train_path = build_output_path(args.output_dir, meter, is_test=False)
        test_path = build_output_path(args.output_dir, meter, is_test=True)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_records, f, ensure_ascii=False, indent=2)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_records, f, ensure_ascii=False, indent=2)

        print(
            f"Train: {len(train_poems):,} poems, {len(train_records):,} couplets -> {train_path}"
        )
        print(
            f"Test:  {len(test_poems):,} poems, {len(test_records):,} couplets -> {test_path}"
        )
        print(
            "Overlap (must be zero): "
            f"poems={overlaps['shared_poem_ids']} "
            f"couplets={overlaps['shared_couplets']} "
            f"lines={overlaps['shared_lines']}"
        )
        if test_shortages or train_shortages:
            print(f"Dynasty shortages: test={test_shortages} train={train_shortages}")

        meter_report = {
            "train_path": train_path,
            "test_path": test_path,
            "available_poems": stats["complete_poems"],
            "train_poems": len(train_poems),
            "test_poems": len(test_poems),
            "train_couplets": len(train_records),
            "test_couplets": len(test_records),
            "train_poems_by_dynasty": dynasty_counts(train_poems),
            "test_poems_by_dynasty": dynasty_counts(test_poems),
            "poems_rejected_for_overlap_with_test": train_rejected,
            "poems_rejected_while_building_test": test_rejected,
            "dynasty_shortages_train": train_shortages,
            "dynasty_shortages_test": test_shortages,
            **overlaps,
        }
        report["meters"][METER_NAMES[meter]] = meter_report

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nWrote split report to {report_path}")
    print("Done!")


if __name__ == "__main__":
    main()
