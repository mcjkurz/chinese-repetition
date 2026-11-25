"""
Fine-tune one character-based causal (GPT) language model on Inner/Outer couplets
of a single meter.

Train pentasyllabic and heptasyllabic verse as two models. Each example is
prefixed with "i" or "o", the same control token the pentasyllabic experiment
uses. The line length stays inside the model, because that model never sees
the other meter.

Training data is in simplified Chinese. GujiGPT expects traditional Chinese,
so the script converts on the fly with OpenCC by default.
"""

import argparse
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import opencc

from common import ensure_padding_token, select_device


DEFAULT_MODEL_NAME = "hsc748NLP/GujiGPT_fan"
DEFAULT_TOKENIZER_NAME = None
DEFAULT_DATA_PATH = "data/2-3_regulated_verse/couplets_penta_train.json"
DEFAULT_OUTPUT_DIR = "models/gujigpt-io-finetuned-penta"
DEFAULT_CONTROL_OUTPUT_DIR = "models/gujigpt-io-finetuned-shuffled-penta"
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_MAX_LENGTH = 24
DEFAULT_SHUFFLE_SEED = 42


def load_couplets(data_path: str) -> list[dict]:
    """Load one meter's couplet corpus and refuse a mixed-meter file."""
    with open(data_path, "r", encoding="utf-8") as f:
        couplets = json.load(f)
    print(f"  Loaded {len(couplets)} couplets from {data_path}")
    meters = {item["meter"] for item in couplets}
    if len(meters) != 1:
        raise SystemExit(
            f"{data_path} contains meters {sorted(meters)}. "
            "Train one meter per model."
        )
    return couplets


class CoupletDataset(Dataset):
    """Dataset for couplets prefixed with i or o."""

    def __init__(
        self,
        couplets: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = DEFAULT_MAX_LENGTH,
        convert_to_traditional: bool = True,
        shuffle_labels: bool = False,
        shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        converter = opencc.OpenCC('s2t') if convert_to_traditional else None
        labels = [item["label"] for item in couplets]
        if shuffle_labels:
            labels = labels.copy()
            rng = random.Random(shuffle_seed)
            rng.shuffle(labels)

        for item, label in zip(couplets, labels):
            # Format: i明月松間照，清泉石上流。 or o風急天高猿嘯哀，渚清沙白鳥飛回。
            couplet_text = item['couplet']

            if converter:
                couplet_text = converter.convert(couplet_text)

            text = label + couplet_text + "。"

            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                add_special_tokens=False,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)

            labels_tensor = input_ids.clone()
            labels_tensor[attention_mask == 0] = -100

            self.examples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels_tensor
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int
):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a character-based causal language model on Inner/Outer classified couplets"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face model name or local path (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=DEFAULT_TOKENIZER_NAME,
        help="Hugging Face tokenizer name or local path (default: same as --model-name)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Maximum tokenized sequence length (default: {DEFAULT_MAX_LENGTH})"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Training couplets for one meter (default: {DEFAULT_DATA_PATH})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save fine-tuned model"
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip simplified→traditional conversion (use if data is already traditional)"
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="Train a shuffled-label control model with randomly permuted i/o prefixes"
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help=f"Random seed for --control label shuffling (default: {DEFAULT_SHUFFLE_SEED})"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model/tokenizer code from Hugging Face repositories"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for torch, random and the batch order (default: none, as for the original models)"
    )
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    output_dir = args.output_dir
    tokenizer_name = args.tokenizer_name or args.model_name
    if args.control and output_dir == DEFAULT_OUTPUT_DIR:
        output_dir = DEFAULT_CONTROL_OUTPUT_DIR

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.epochs
    max_length = args.max_length

    device = select_device()
    print(f"Using device: {device}")

    print("Loading classified couplets...")
    couplets = load_couplets(args.data_path)
    print(f"Loaded {len(couplets)} couplets, meter {couplets[0]['meter']}")
    if args.control:
        print(
            "Control mode: randomly shuffling i/o labels across couplets "
            f"(seed: {args.shuffle_seed})"
        )

    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=args.trust_remote_code,
    )
    ensure_padding_token(tokenizer)

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    convert_to_traditional = not args.no_convert
    print(f"Creating dataset... (convert to traditional: {convert_to_traditional})")
    dataset = CoupletDataset(
        couplets,
        tokenizer,
        max_length=max_length,
        convert_to_traditional=convert_to_traditional,
        shuffle_labels=args.control,
        shuffle_seed=args.shuffle_seed,
    )
    generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    print(f"\nStarting training for {num_epochs} epoch(s)...")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total steps: {total_steps}")

    for epoch in range(num_epochs):
        avg_loss = train(model, dataloader, optimizer, scheduler, device, epoch)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
