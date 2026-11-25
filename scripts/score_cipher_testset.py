"""Perplexity and prompt effects of a fine-tuned GujiGPT on a held-out test set, for the decoding controls table.

Scores every couplet of --test-file (the 5,000 inner and 5,000 outer held-out
couplets of a meter) under the inner (i) and the outer (o) prompt with
score_couplet_controls.score_set: at every position of the second line, the
loss of the actual character given the prompt, the first line, the comma and
the true preceding characters. Reports, as the table of the article defines
them, the perplexity under the aligned prompt (exp of the mean token loss of
inner couplets under I and of outer couplets under O) and the prompt effect
for each couplet type (mean loss under the matched minus under the mismatched
prompt, in nats). --convert converts simplified couplets to traditional (the
real test set); the enciphered test sets of encipher_testset.py are already
traditional.

Appends one row per call to --output (results/cipher_finetune/table2_rows.json).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import select_device
from score_couplet_controls import load, score_set


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--meter", choices=["penta", "hepta"], required=True)
    parser.add_argument("--label", required=True, help="Row name, e.g. 'nonsense epoch 1'")
    parser.add_argument("--convert", action="store_true", help="Convert simplified couplets to traditional")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="results/cipher_finetune/table2_rows.json")
    args = parser.parse_args()

    device = select_device()
    model, tokenizer = load(args.model, device)
    items = json.loads(Path(args.test_file).read_text(encoding="utf-8"))
    for i, item in enumerate(items):
        item.setdefault("id", f"{args.meter}-test-{i:05d}")
    by_type = {t: [x for x in items if x["label"] == t] for t in ("i", "o")}
    loss = {}
    for prompt in ("i", "o"):
        for t, subset in by_type.items():
            rows = score_set(subset, model, tokenizer, prompt, device, args.batch_size, convert=args.convert)
            loss[(prompt, t)] = float(np.mean([v for r in rows for v in r["loss"]]))
            print(f"{args.label} {args.meter} prompt {prompt} couplets {t}: mean loss {loss[(prompt, t)]:.4f}",
                  flush=True)
    row = {"label": args.label, "meter": args.meter, "model": args.model, "test_file": args.test_file,
           "n": {t: len(v) for t, v in by_type.items()},
           "mean_loss": {f"{p}/{t}": v for (p, t), v in loss.items()},
           "ppl_inner": math.exp(loss[("i", "i")]), "ppl_outer": math.exp(loss[("o", "o")]),
           "prompt_effect_inner": loss[("i", "i")] - loss[("o", "i")],
           "prompt_effect_outer": loss[("o", "o")] - loss[("i", "o")]}
    print(f"{args.label} {args.meter}: inner PPL {row['ppl_inner']:.1f}, outer PPL {row['ppl_outer']:.1f}, "
          f"prompt effect inner {row['prompt_effect_inner']:+.2f}, outer {row['prompt_effect_outer']:+.2f}", flush=True)
    output = Path(args.output)
    rows = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    rows = [r for r in rows if not (r["label"] == args.label and r["meter"] == args.meter)] + [row]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
