"""Fine-tune GujiBERT to predict second-line characters from an all-mask line."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from bert_io_common import MaskedCoupletDataset, load_couplets
from common import METER_NAMES, select_device


DEFAULT_MODEL_NAME = "hsc748NLP/GujiBERT_fan"
DEFAULT_DATA_PATH = "data/2-3_regulated_verse/couplets_penta_train.json"
DEFAULT_OUTPUT_DIR = "models/gujibert-io-masked-finetuned-penta"
DEFAULT_CONTROL_OUTPUT_DIR = "models/gujibert-io-masked-finetuned-shuffled-penta"


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    for batch in progress:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.item())
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(dataloader)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune masked GujiBERT on inner/outer couplets, predicting "
            "each second-line character while the whole second line is masked."
        )
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-training-samples",
        type=int,
        help="Optional limit for smoke tests; omit to use all training couplets.",
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip simplified-to-traditional conversion.",
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="Shuffle inner/outer labels before fine-tuning.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    couplets = load_couplets(args.data_path)
    meter = couplets[0]["meter"]
    if meter not in METER_NAMES:
        raise ValueError(f"Unsupported meter: {meter}")
    if args.max_training_samples is not None:
        if args.max_training_samples < 1:
            raise ValueError("--max-training-samples must be positive.")
        couplets = couplets[: args.max_training_samples]
    if args.control and args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = DEFAULT_CONTROL_OUTPUT_DIR.replace(
            "penta", METER_NAMES[meter]
        )
    elif args.output_dir == DEFAULT_OUTPUT_DIR and meter == 7:
        args.output_dir = DEFAULT_OUTPUT_DIR.replace("penta", "hepta")

    device = select_device()
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = MaskedCoupletDataset(
        couplets,
        tokenizer,
        meter,
        convert_to_traditional=not args.no_convert,
        shuffle_labels=args.control,
        shuffle_seed=args.shuffle_seed,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    for epoch in range(args.epochs):
        mean_loss = train_epoch(
            model, dataloader, optimizer, scheduler, device, epoch
        )
        print(f"Epoch {epoch + 1} mean masked-character loss: {mean_loss:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "base_model": args.model_name,
        "task": "masked_second_line_character_prediction",
        "input_template": (
            "[CLS] {i_or_o}{first_line}，"
            + tokenizer.mask_token * meter
            + "[SEP]"
        ),
        "loss_positions": "all masked second-line character positions",
        "meter": meter,
        "training_examples": len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "shuffled_labels": args.control,
        "shuffle_seed": args.shuffle_seed if args.control else None,
        "seed": args.seed,
    }
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"Saved fine-tuned model and experiment_config.json to {output_dir}")


if __name__ == "__main__":
    main()
