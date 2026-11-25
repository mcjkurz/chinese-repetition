"""Fine-tune GujiGPT on enciphered (or real) regulated-verse couplets and score it after every epoch.

Standalone: needs only couplets.json in the same directory and the packages
torch, transformers and numpy (pip install torch transformers numpy).

For every meter in --meters and corpus in --corpora ("cipher" by default;
"real" trains on the same couplets unenciphered, as a reference with the same
schedule), the script

1. scores the pretrained GujiGPT (epoch 0) on the held-out inner and outer
   couplets of that corpus, each under the inner (i) and the outer (o) prompt;
2. fine-tunes it for --epochs epochs on the 100,000 training couplets, each
   written as its control token, the couplet and "。" (e.g. "i明月松間照，清泉石上流。"),
   with AdamW, a linear decay without warmup and gradient clipping at 1.0, as
   the models of the article were trained (they used one epoch at lr 2e-5,
   batch size 8);
3. after every epoch, scores the model again in the same way.

Scoring follows the article: at every position of the second line, the model
receives the control token, the first line, the comma and the true preceding
characters of the second line, and the entropy of its next-character
distribution and the surprisal of the actual character are recorded (in nats).

Writes, under --output-dir:
- {meter}_{corpus}_epoch{k}.json: per-couplet entropy and surprisal at every
  position, for the keys "I/inner", "O/inner", "I/outer", "O/outer"
  (prompt/test set), with the training loss of the epoch;
- summary.md and summary.json: mean entropy and surprisal by position for
  every meter, corpus, epoch, prompt and test set (rewritten after every epoch).
With --save-every-epoch, the model after every epoch goes to
--model-dir/model_{meter}_{corpus}_epoch{k}. Upload the output directory
(without the models).

Examples:
    python finetune_cipher.py                                  # both meters, cipher, 4 epochs
    python finetune_cipher.py --meters penta --epochs 4 --bf16
    python finetune_cipher.py --corpora cipher real            # also the real-text reference
    python finetune_cipher.py --limit-train 64 --limit-test 8 --epochs 1   # smoke test
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
PROMPTS = {"I": "i", "O": "o"}


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if enabled else contextlib.nullcontext()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(name)
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model.to(device), tokenizer


def training_tensors(couplets: list[dict], tokenizer, max_length: int) -> TensorDataset:
    texts = [item["label"] + item["couplet"] + "。" for item in couplets]
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length",
                    add_special_tokens=False, return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return TensorDataset(enc["input_ids"], enc["attention_mask"], labels)


def train_epoch(model, loader, optimizer, scheduler, device, amp: bool, log_every: int, tag: str) -> float:
    model.train()
    total, steps, start = 0.0, 0, time.time()
    for input_ids, attention_mask, labels in loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        with autocast(device, amp):
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total += loss.item()
        steps += 1
        if log_every and steps % log_every == 0:
            rate = steps / (time.time() - start)
            print(f"  {tag} step {steps}/{len(loader)}  loss {total / steps:.4f}  "
                  f"({rate:.1f} steps/s, {(len(loader) - steps) / rate / 60:.1f} min left)", flush=True)
    return total / max(steps, 1)


@torch.no_grad()
def score(model, tokenizer, items: list[dict], prompt: str, device, batch_size: int, amp: bool) -> list[dict]:
    """Entropy and surprisal at every position of the second line, per couplet (teacher forcing)."""
    model.eval()
    examples, owners = [], []
    for index, item in enumerate(items):
        first, second = item["couplet"].split("，")
        for k in range(len(second)):
            prefix = tokenizer(prompt + first + "，" + second[:k], add_special_tokens=False)["input_ids"]
            target = tokenizer(second[k], add_special_tokens=False)["input_ids"]
            examples.append((prefix, target))
            owners.append(index)
    entropies, losses = [], []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        width = max(len(p) + len(t) for p, t in batch)
        ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, (p, t) in enumerate(batch):
            ids[row, :len(p) + len(t)] = torch.tensor(p + t)
            mask[row, :len(p) + len(t)] = 1
        ids, mask = ids.to(device), mask.to(device)
        with autocast(device, amp):
            logits = model(input_ids=ids, attention_mask=mask).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)
        rows = torch.arange(len(batch), device=device)
        next_pos = torch.tensor([len(p) - 1 for p, _ in batch], device=device)
        next_lp = log_probs[rows, next_pos]
        entropies.extend((-(next_lp.exp() * next_lp).sum(-1)).tolist())
        for row, (p, t) in enumerate(batch):
            # Surprisal of the target, averaged over its tokens if it splits into several.
            positions = torch.arange(len(p) - 1, len(p) + len(t) - 1, device=device)
            target_ids = ids[row, len(p):len(p) + len(t)]
            losses.append(float(-log_probs[row, positions, target_ids].mean()))
    out = [{"id": item["id"], "entropy": [], "loss": []} for item in items]
    for owner, e, l in zip(owners, entropies, losses):
        out[owner]["entropy"].append(round(e, 4))
        out[owner]["loss"].append(round(l, 4))
    return out


def evaluate(model, tokenizer, tests: dict, device, args) -> dict:
    scores = {}
    for prompt_name, prompt in PROMPTS.items():
        for test_name, items in tests.items():
            scores[f"{prompt_name}/{test_name}"] = score(model, tokenizer, items, prompt, device,
                                                         args.score_batch_size, args.bf16)
    return scores


def profile(rows: list[dict], key: str) -> list[float]:
    return [round(float(v), 4) for v in np.array([r[key] for r in rows]).mean(axis=0)]


def write_summary(summary: list[dict], out_dir: Path) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    lines = ["| Meter | Corpus | Epoch | Prompt/test | Entropy by position | Mean entropy | Mean surprisal | Train loss |",
             "|:--|:--|--:|:--|:--|--:|--:|--:|"]
    for row in summary:
        loss = "—" if row["train_loss"] is None else f"{row['train_loss']:.3f}"
        lines.append(f"| {row['meter']} | {row['corpus']} | {row['epoch']} | {row['key']} | "
                     f"{' '.join(f'{v:.2f}' for v in row['entropy'])} | {np.mean(row['entropy']):.3f} | "
                     f"{np.mean(row['loss']):.3f} | {loss} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(HERE / "couplets.json"))
    parser.add_argument("--meters", nargs="+", choices=["penta", "hepta"], default=["penta", "hepta"])
    parser.add_argument("--corpora", nargs="+", choices=["cipher", "real"], default=["cipher"])
    parser.add_argument("--model", default="hsc748NLP/GujiGPT_fan")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=24)
    parser.add_argument("--score-batch-size", type=int, default=256, help="Prefixes per scoring batch")
    parser.add_argument("--bf16", action="store_true", help="bfloat16 autocast (CUDA), for speed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1000, help="Training steps between progress lines")
    parser.add_argument("--save-model", action="store_true", help="Also save the final model of every run")
    parser.add_argument("--save-every-epoch", action="store_true",
                        help="Save the model after every epoch (about 450 MB each) under --model-dir")
    parser.add_argument("--model-dir", default=None, help="Where to save models (default: --output-dir)")
    parser.add_argument("--limit-train", type=int, default=None, help="First N training couplets (smoke tests)")
    parser.add_argument("--limit-test", type=int, default=None, help="First N test couplets per set (smoke tests)")
    parser.add_argument("--output-dir", default=str(HERE / "results"))
    args = parser.parse_args()

    device = select_device()
    print(f"Device: {device}; config: {vars(args)}", flush=True)
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))["meters"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir) if args.model_dir else out_dir
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []

    for meter in args.meters:
        for corpus in args.corpora:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            part = data[meter][corpus]
            train = part["train"][:args.limit_train] if args.limit_train else part["train"]
            tests = {k: (part[k][:args.limit_test] if args.limit_test else part[k]) for k in ("inner", "outer")}
            model, tokenizer = load_model(args.model, device)
            loader = DataLoader(training_tensors(train, tokenizer, args.max_length), batch_size=args.batch_size,
                                shuffle=True, generator=torch.Generator().manual_seed(args.seed))
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(loader) * args.epochs)
            summary = [r for r in summary if not (r["meter"] == meter and r["corpus"] == corpus)]
            print(f"\n== {meter} {corpus}: {len(train)} training couplets, {len(loader)} steps per epoch", flush=True)

            for epoch in range(args.epochs + 1):
                train_loss = None
                if epoch > 0:
                    train_loss = train_epoch(model, loader, optimizer, scheduler, device, args.bf16,
                                             args.log_every, f"{meter} {corpus} epoch {epoch}")
                scores = evaluate(model, tokenizer, tests, device, args)
                result = {"meter": meter, "corpus": corpus, "epoch": epoch, "train_loss": train_loss,
                          "config": {k: v for k, v in vars(args).items() if k not in ("meters", "corpora")},
                          "scores": scores}
                (out_dir / f"{meter}_{corpus}_epoch{epoch}.json").write_text(
                    json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
                for key, rows in scores.items():
                    summary.append({"meter": meter, "corpus": corpus, "epoch": epoch, "key": key,
                                    "train_loss": train_loss, "entropy": profile(rows, "entropy"),
                                    "loss": profile(rows, "loss")})
                    if key in ("I/inner", "O/outer"):
                        print(f"  {meter} {corpus} epoch {epoch} {key}: entropy by position "
                              f"{' '.join(f'{v:.2f}' for v in summary[-1]['entropy'])}, "
                              f"surprisal {np.mean(summary[-1]['loss']):.3f}", flush=True)
                write_summary(summary, out_dir)
                if args.save_every_epoch and epoch > 0:
                    path = model_dir / f"model_{meter}_{corpus}_epoch{epoch}"
                    model.save_pretrained(path)
                    tokenizer.save_pretrained(path)
                    print(f"  saved {path}", flush=True)
            if args.save_model:
                model.save_pretrained(model_dir / f"model_{meter}_{corpus}")
                tokenizer.save_pretrained(model_dir / f"model_{meter}_{corpus}")
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"\nDone. Results in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
