"""Train GujiBERT with a new lexicon on the enciphered corpus: MLM, word embeddings only, body frozen.

The word embeddings (tied to the MLM decoder) are re-initialized and trained with the decoder bias; every other
parameter of GujiBERT (hsc748NLP/GujiBERT_fan) stays frozen ("same reader, new lexicon"). Input: the output of
prepare.py. A fixed set of --heldout sequences is never trained on; held-out MLM loss on it is logged every
--eval-every steps, with GujiBERT's own loss on the same sequences in real characters as the reference.

Writes, under --out: log.jsonl (one line per evaluation), embeddings_step{N}.pt and embeddings_final.pt (the
trained parameters only), summary.json.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, BertForMaskedLM

MODEL = "hsc748NLP/GujiBERT_fan"
ROOT = Path(__file__).resolve().parents[2]
TRAINABLE = ("bert.embeddings.word_embeddings.weight", "cls.predictions.bias")


def chunks(ids: np.ndarray, seq_len: int, cls: int, sep: int) -> torch.Tensor:
    body = seq_len - 2
    n = len(ids) // body
    x = ids[: n * body].reshape(n, body).astype(np.int64)
    return torch.from_numpy(np.concatenate([np.full((n, 1), cls), x, np.full((n, 1), sep)], axis=1))


def mask_batch(x, tok, gen, special):
    labels = x.clone()
    prob = torch.full(x.shape, 0.15)
    prob[torch.isin(x, special)] = 0
    masked = torch.bernoulli(prob, generator=gen).bool()
    labels[~masked] = -100
    inputs = x.clone()
    replace = torch.bernoulli(torch.full(x.shape, 0.8), generator=gen).bool() & masked
    inputs[replace] = tok.mask_token_id
    rand = torch.bernoulli(torch.full(x.shape, 0.5), generator=gen).bool() & masked & ~replace
    inputs[rand] = torch.randint(len(tok), x.shape, generator=gen)[rand]
    return inputs, labels


@torch.no_grad()
def heldout_loss(model, data, tok, special, device, batch=64) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(1234)  # the same masks at every evaluation
    total, count = 0.0, 0
    for i in range(0, len(data), batch):
        inputs, labels = mask_batch(data[i:i + batch], tok, gen, special)
        out = model(input_ids=inputs.to(device), labels=labels.to(device))
        n = (labels != -100).sum().item()
        total += out.loss.item() * n
        count += n
    model.train()
    return total / count


def save(model, path: Path) -> None:
    state = dict(model.named_parameters())
    torch.save({name: state[name].detach().cpu() for name in TRAINABLE}, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=str(ROOT / "results/cipher_bert/corpus"))
    p.add_argument("--out", default=str(ROOT / "results/cipher_bert/frozen_body"))
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--heldout", type=int, default=512)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    special = torch.tensor([tok.cls_token_id, tok.sep_token_id, tok.pad_token_id])
    real = chunks(np.load(Path(args.corpus) / "real_ids.npy"), args.seq_len, tok.cls_token_id, tok.sep_token_id)
    ciph = chunks(np.load(Path(args.corpus) / "cipher_ids.npy"), args.seq_len, tok.cls_token_id, tok.sep_token_id)
    order = torch.randperm(len(ciph), generator=torch.Generator().manual_seed(args.seed))
    held, train = order[: args.heldout], order[args.heldout:]
    steps_per_epoch = len(train) // args.batch
    total_steps = int(args.epochs * steps_per_epoch)
    log = open(out / "log.jsonl", "a", encoding="utf-8")

    def write(record):
        record["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(record, flush=True)

    reference = BertForMaskedLM.from_pretrained(MODEL).to(device)
    write({"event": "reference", "gujibert_real_heldout_loss": heldout_loss(reference, real[held], tok, special, device),
           "gujibert_cipher_heldout_loss": heldout_loss(reference, ciph[held], tok, special, device),
           "train_sequences": len(train), "steps_per_epoch": steps_per_epoch, "total_steps": total_steps})
    del reference
    if device.type == "mps":
        torch.mps.empty_cache()

    model = BertForMaskedLM.from_pretrained(MODEL)
    torch.nn.init.normal_(model.bert.embeddings.word_embeddings.weight, std=model.config.initializer_range)
    model.tie_weights()
    for name, param in model.named_parameters():
        param.requires_grad = name in TRAINABLE
    trainable = [q for q in model.parameters() if q.requires_grad]
    model.to(device).train()
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    def schedule(step):  # linear warm-up, then linear decay to a tenth of the peak
        if step < args.warmup:
            return (step + 1) / args.warmup
        return max(0.1, 1 - 0.9 * (step - args.warmup) / max(1, total_steps - args.warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    write({"event": "start", "step": 0, "heldout_loss": heldout_loss(model, ciph[held], tok, special, device),
           "args": vars(args)})
    gen = torch.Generator().manual_seed(args.seed + 1)
    start, running = time.time(), []
    step = 0
    while step < total_steps:
        perm = train[torch.randperm(len(train), generator=gen)]
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            inputs, labels = mask_batch(ciph[perm[i:i + args.batch]], tok, gen, special)
            loss = model(input_ids=inputs.to(device), labels=labels.to(device)).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            running.append(loss.item())
            if step % args.eval_every == 0 or step == total_steps:
                elapsed = time.time() - start
                write({"event": "eval", "step": step, "epoch": round(step / steps_per_epoch, 3),
                       "train_loss": round(float(np.mean(running)), 4),
                       "heldout_loss": round(heldout_loss(model, ciph[held], tok, special, device), 4),
                       "lr": sched.get_last_lr()[0], "hours_elapsed": round(elapsed / 3600, 3),
                       "hours_left": round(elapsed / step * (total_steps - step) / 3600, 2)})
                running = []
            if step % args.save_every == 0:
                save(model, out / f"embeddings_step{step}.pt")
            if step == total_steps:
                break
    save(model, out / "embeddings_final.pt")
    summary = {"steps": step, "hours": round((time.time() - start) / 3600, 2),
               "final_heldout_loss": heldout_loss(model, ciph[held], tok, special, device)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    write({"event": "done", **summary})


if __name__ == "__main__":
    main()
