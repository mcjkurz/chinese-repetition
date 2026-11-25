"""Rank inner couplets by transition alignment.

This experiment reads a prepared alignment JSON (penta or hepta), keeps only
position 2 and 3 (inner) couplets, computes the same transition alignment as
transition_analysis.py, and saves couplets sorted from most to least aligned
on one chosen layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import opencc
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from common import select_device, split_couplet
from transition_analysis import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_LABEL,
    DEFAULT_TOKENIZER,
    LAYERS,
    couplet_context_and_targets,
    get_character_embeddings_batch,
    iter_batches,
    layer_transition_alignment,
    projection_description,
    token_count_matches_text_length,
)


DEFAULT_DATA_PATH = "data/2-3_regulated_verse/couplets_penta_alignment_tang.json"
DEFAULT_OUTPUT_PATH = (
    "results/transition_alignment_examples/"
    "ranked_transition_alignment_tang_positions_2_3.json"
)
TARGET_POSITIONS = (2, 3)


def normalize_layer_index(layer_index: int) -> int:
    """Normalize positive and negative layer indexes against available layers."""
    layer_count = len(LAYERS)
    if layer_index < 0:
        layer_index = layer_count + layer_index
    if not 0 <= layer_index < layer_count:
        raise ValueError(
            f"--rank-layer must be in [0, {layer_count - 1}] "
            "or a valid negative index"
        )
    return layer_index


def traditional_lines(item: dict, converter: opencc.OpenCC) -> tuple[str | None, str | None]:
    """Return traditional line text, converting simplified fields as needed."""
    line1 = item.get("line1_traditional")
    line2 = item.get("line2_traditional")
    if line1 and line2:
        return line1, line2

    couplet = item.get("couplet_traditional")
    if couplet:
        line1, line2 = split_couplet(couplet)
        if line1 and line2:
            return line1, line2

    line1 = item.get("line1")
    line2 = item.get("line2")
    if line1 and line2:
        return converter.convert(line1), converter.convert(line2)

    couplet = item.get("couplet")
    if couplet:
        line1, line2 = split_couplet(couplet)
        if line1 and line2:
            return converter.convert(line1), converter.convert(line2)

    return None, None


def normalize_record(item: dict, converter: opencc.OpenCC) -> dict | None:
    """Return one clean position-2/3 record with traditional line fields."""
    position = item.get("position")
    if position not in TARGET_POSITIONS:
        return None

    line1_traditional, line2_traditional = traditional_lines(item, converter)
    if (
        not line1_traditional
        or not line2_traditional
        or len(line1_traditional) != len(line2_traditional)
        or len(line1_traditional) < 2
    ):
        return None

    couplet_traditional = f"{line1_traditional}，{line2_traditional}"
    return {
        "couplet": item.get("couplet", couplet_traditional),
        "couplet_traditional": item.get("couplet_traditional", couplet_traditional),
        "line1": item.get("line1"),
        "line2": item.get("line2"),
        "line1_traditional": line1_traditional,
        "line2_traditional": line2_traditional,
        "position": position,
    }


def load_inner_couplets(data_path: Path, limit: int | None = None) -> tuple[list[dict], int]:
    """Load all position-2/3 (inner) couplets from the alignment JSON."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Alignment data not found: {data_path}. "
            "Create it first with scripts/prepare_data_alignment.py."
        )

    converter = opencc.OpenCC("s2t")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    malformed = 0
    for item in data:
        if item.get("position") not in TARGET_POSITIONS:
            continue
        record = normalize_record(item, converter)
        if record is None:
            malformed += 1
            continue
        records.append(record)
        if limit is not None and len(records) >= limit:
            break

    if malformed:
        print(f"Skipped {malformed} malformed position-2/3 records while loading")
    return records, malformed


def validate_records(
    records: list[dict],
    tokenizer,
    include_punctuation: bool,
) -> tuple[list[dict], list[str], list[list[int]], int]:
    """Keep records whose encoder context maps one character to one token."""
    valid_records = []
    texts = []
    target_indices_batch = []
    skipped = 0

    for record in records:
        text, target_indices = couplet_context_and_targets(
            record["line1_traditional"],
            record["line2_traditional"],
            include_punctuation,
        )
        if not token_count_matches_text_length(text, tokenizer):
            skipped += 1
            continue
        valid_records.append(record)
        texts.append(text)
        target_indices_batch.append(target_indices)

    return valid_records, texts, target_indices_batch, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank position-2/3 (inner) couplets by per-layer transition alignment"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="Tokenizer path or Hugging Face identifier (default: same as --model)",
    )
    parser.add_argument("--model-label", type=str, default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--data-path", type=Path, default=Path(DEFAULT_DATA_PATH))
    parser.add_argument("--output-path", type=Path, default=Path(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--rank-layer",
        type=int,
        default=-1,
        help="Layer used for sorting, with -1 meaning the final layer",
    )
    parser.add_argument(
        "--pca-dims",
        "--PCA",
        dest="pca_dims",
        type=int,
        default=None,
        help="Optional per-couplet PCA projection before alignment",
    )
    parser.add_argument(
        "--no-punctuation",
        action="store_true",
        help="Encode each couplet as line1+line2 instead of line1，line2。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for quick test runs; defaults to all position-2/3 records",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")
    if args.pca_dims is not None and args.pca_dims <= 0:
        raise ValueError("--pca-dims/--PCA must be a positive integer")

    rank_layer = normalize_layer_index(args.rank_layer)
    include_punctuation = not args.no_punctuation
    tokenizer_name = args.tokenizer or args.model
    projection = projection_description(args.pca_dims)

    records, malformed_count = load_inner_couplets(args.data_path, args.limit)
    print(f"Loaded {len(records)} position-2/3 (inner) couplets")
    print(f"Ranking by layer {rank_layer} in {projection}")

    device = select_device()
    print(f"Using device: {device}")
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Loading {args.model_label} encoder: {args.model}")
    model = AutoModel.from_pretrained(
        args.model,
        output_hidden_states=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    ranked_records = []
    skipped = 0
    for batch in tqdm(iter_batches(records, args.batch_size), desc="Batches"):
        batch, texts, target_indices_batch, token_skips = validate_records(
            batch,
            tokenizer,
            include_punctuation,
        )
        skipped += token_skips
        if not batch:
            continue

        embeddings_batch = get_character_embeddings_batch(
            texts,
            target_indices_batch,
            tokenizer,
            model,
            device,
        )
        for record, embeddings_by_layer in zip(batch, embeddings_batch):
            line_len = len(record["line1_traditional"])
            if embeddings_by_layer.shape[1] < line_len * 2:
                skipped += 1
                continue

            alignment = layer_transition_alignment(
                embeddings_by_layer[:, :line_len, :],
                embeddings_by_layer[:, line_len:line_len * 2, :],
                args.pca_dims,
            )
            ranked_records.append(
                {
                    "couplet": record["couplet_traditional"],
                    "transition_alignment_score": alignment["layer_mean_alignments"][rank_layer],
                }
            )

    ranked_records.sort(
        key=lambda record: record["transition_alignment_score"],
        reverse=True,
    )

    payload = {
        "data_path": str(args.data_path),
        "target_positions": list(TARGET_POSITIONS),
        "layers": list(LAYERS),
        "rank_layer": rank_layer,
        "requested_rank_layer": args.rank_layer,
        "pca_dims": args.pca_dims,
        "include_punctuation": include_punctuation,
        "alignment_space": projection,
        "model_name": args.model,
        "tokenizer_name": tokenizer_name,
        "model_label": args.model_label,
        "record_count": len(ranked_records),
        "malformed_count": malformed_count,
        "tokenization_or_embedding_skipped_count": skipped,
        "ranked_couplets": ranked_records,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Skipped {skipped} malformed or non-character-tokenized records")
    print(f"Saved ranked transition alignment: {args.output_path}")


if __name__ == "__main__":
    main()
