"""
Filter the raw poem CSVs down to regulated verse (律詩) and consolidate them
into a single CSV.

We keep only rows whose 体裁 (genre) is exactly "五言律诗" (pentasyllabic
regulated verse) or "七言律诗" (heptasyllabic regulated verse) - this
excludes the related but structurally different forms 五言排律/七言排律
(extended regulated verse, more than four couplets) and 五言绝句/七言绝句
(quatrains, only two couplets).

Dynasties are consolidated to the five the downstream experiment balances
across (唐, 宋, 元, 明, 清); the source repo splits populous dynasties like
宋/明/清 into several files (宋_1.csv, 宋_2.csv, ...), but the 朝代 column
inside each row already carries the plain dynasty name, so no renaming is
needed beyond filtering.
"""

import argparse
import csv
import glob
import os
from collections import Counter

from common import METER_NAMES

DYNASTIES = ("唐", "宋", "元", "明", "清")
GENRE_TO_METER = {"五言律诗": 5, "七言律诗": 7}
DEFAULT_INPUT_DIR = "data/2-3_regulated_verse/raw"
DEFAULT_OUTPUT_PATH = "data/2-3_regulated_verse/regulated.csv"
OUTPUT_FIELDNAMES = ["dynasty", "meter", "title", "author", "content"]


def iter_source_rows(input_dir: str):
    """Yield rows from every CSV in input_dir, skipping malformed ones."""
    paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in {input_dir}. Run fetch_data.py first."
        )

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row


def main():
    parser = argparse.ArgumentParser(
        description="Filter raw poem CSVs to pentasyllabic/heptasyllabic regulated verse"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory of raw source CSVs (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    stats = Counter()
    kept_rows = []

    print(f"Reading raw CSVs from {args.input_dir} ...")
    for row in iter_source_rows(args.input_dir):
        stats["total_rows"] += 1

        dynasty = (row.get("朝代") or "").strip()
        genre = (row.get("体裁") or "").strip()
        content = (row.get("内容") or "").strip()

        if dynasty not in DYNASTIES:
            stats["skipped_dynasty"] += 1
            continue
        if genre not in GENRE_TO_METER:
            stats["skipped_genre"] += 1
            continue
        if not content:
            stats["skipped_empty_content"] += 1
            continue

        meter = GENRE_TO_METER[genre]
        kept_rows.append(
            {
                "dynasty": dynasty,
                "meter": meter,
                "title": (row.get("标题") or "").strip(),
                "author": (row.get("作者") or "").strip(),
                "content": content,
            }
        )
        stats[f"kept_{dynasty}_{METER_NAMES[meter]}"] += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"\nTotal source rows scanned: {stats['total_rows']:,}")
    print(f"Skipped (unsupported dynasty): {stats['skipped_dynasty']:,}")
    print(f"Skipped (not 五言律诗/七言律诗): {stats['skipped_genre']:,}")
    print(f"Skipped (empty content): {stats['skipped_empty_content']:,}")
    print(f"\nKept {len(kept_rows):,} regulated poems, by dynasty and meter:")
    for dynasty in DYNASTIES:
        for meter in (5, 7):
            key = f"kept_{dynasty}_{METER_NAMES[meter]}"
            print(f"  {dynasty} {METER_NAMES[meter]} ({meter}言律诗): {stats[key]:,}")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
