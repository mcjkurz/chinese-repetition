"""Shared data preparation for the masked-second-line GujiBERT experiment."""

from __future__ import annotations

import json

import opencc
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from common import split_couplet


def load_couplets(data_path: str) -> list[dict]:
    """Load one meter's couplets and reject mixed-meter input."""
    with open(data_path, "r", encoding="utf-8") as file:
        couplets = json.load(file)
    if not couplets:
        raise ValueError(f"No couplets found in {data_path}")
    meters = {item["meter"] for item in couplets}
    if len(meters) != 1:
        raise ValueError(
            f"{data_path} contains meters {sorted(meters)}; train one meter per model."
        )
    return couplets


def encode_masked_couplet(
    tokenizer: PreTrainedTokenizerBase,
    couplet: str,
    label: str,
    meter: int,
    converter: opencc.OpenCC | None,
) -> tuple[list[int], list[int], list[int]]:
    """Encode the prompt and first line with every second-line position masked."""
    if label not in ("i", "o"):
        raise ValueError(f"Expected an i/o label, got {label!r}")

    line1, line2 = split_couplet(couplet)
    if line1 is None or line2 is None:
        raise ValueError(f"Malformed couplet: {couplet!r}")
    if converter is not None:
        line1 = converter.convert(line1)
        line2 = converter.convert(line2)
    if len(line1) != meter or len(line2) != meter:
        raise ValueError(
            f"Expected {meter} characters per line, got "
            f"{len(line1)} and {len(line2)} in {couplet!r}"
        )

    prompt_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    first_line_ids = tokenizer(line1, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(line2, add_special_tokens=False)["input_ids"]
    comma_ids = tokenizer("，", add_special_tokens=False)["input_ids"]
    if len(prompt_ids) != 1:
        raise ValueError(f"Prompt {label!r} must tokenize as one token: {prompt_ids}")
    if len(first_line_ids) != meter or len(target_ids) != meter:
        raise ValueError(
            "The tokenizer must represent each Chinese character as one token; "
            f"got {len(first_line_ids)} and {len(target_ids)} tokens for "
            f"{meter}-character lines."
        )
    if len(comma_ids) != 1:
        raise ValueError(f"Chinese comma must tokenize as one token: {comma_ids}")
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("GujiBERT tokenizer must define CLS and SEP tokens.")
    if tokenizer.mask_token_id is None:
        raise ValueError("GujiBERT tokenizer must define a mask token.")

    mask_start = 1 + len(prompt_ids) + len(first_line_ids) + len(comma_ids)
    input_ids = (
        [tokenizer.cls_token_id]
        + prompt_ids
        + first_line_ids
        + comma_ids
        + [tokenizer.mask_token_id] * meter
        + [tokenizer.sep_token_id]
    )
    return input_ids, target_ids, list(range(mask_start, mask_start + meter))


class MaskedCoupletDataset(Dataset):
    """BERT inputs with the entire second line masked and loss only on it."""

    def __init__(
        self,
        couplets: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        meter: int,
        convert_to_traditional: bool = True,
        shuffle_labels: bool = False,
        shuffle_seed: int = 42,
    ):
        self.examples = []
        converter = opencc.OpenCC("s2t") if convert_to_traditional else None
        labels = [item["label"] for item in couplets]
        if shuffle_labels:
            import random

            random.Random(shuffle_seed).shuffle(labels)

        for item, label in zip(couplets, labels):
            input_ids, target_ids, mask_positions = encode_masked_couplet(
                tokenizer, item["couplet"], label, meter, converter
            )
            labels_tensor = torch.full((len(input_ids),), -100, dtype=torch.long)
            labels_tensor[mask_positions] = torch.tensor(target_ids, dtype=torch.long)
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                    "labels": labels_tensor,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]
