"""Does a model fill the second conclusion's slots with the new items or copy the first conclusion?

In the Mengzi 6A10 template, the second conclusion (舍生而取義者也) names the
items of its own half: position 57 repeats position 33, and position 60 repeats
position 41. Copying the first conclusion (舍魚而取熊掌者也) would instead put
the item of position 1 at 57 and that of position 8 at 60. For each passage
and slot, the model reads the text up to the slot and we record the surprisal
of the new item, that of the copied item, and which of the two (if either) is
its top prediction. Qwen reads the simplified text (OpenCC t2s), as in
model_comparison.py; a candidate's probability is that of its first token.

Writes results/surprisal/slot_copy_check.json and .md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import opencc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import select_device
from sequence_surprisal import write_json


MODELS = ["hsc748NLP/GujiGPT_fan", "Qwen/Qwen3-8B-Base"]
CONDITIONS = {
    "mengzi": "data/1_mengzi_prose/mengzi_surprisal.txt",
    "composed": "data/1_mengzi_prose/surprisal_controls/sensical.json",
    "nonsense (items random, composed frame)": "data/1_mengzi_prose/surprisal_controls/items_random_composed_frame.json",
    "items random, Mengzi frame": "data/1_mengzi_prose/surprisal_controls/items_random_mengzi_frame.json",
    "frame and items random": "data/1_mengzi_prose/surprisal_controls/nonsense.json",
}
# Slot (1-based) -> (position of the new item, position of the copied item).
SLOTS = {57: (33, 1), 60: (41, 8)}


def load(path: str) -> list[str]:
    if path.endswith(".txt"):
        return [Path(path).read_text(encoding="utf-8").strip()]
    return [p["text"] for p in json.loads(Path(path).read_text(encoding="utf-8"))["passages"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="results/surprisal/slot_copy_check.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    texts = {name: load(path) for name, path in CONDITIONS.items()}
    rows = []
    for model_name in MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device).eval()
        converter = opencc.OpenCC("t2s") if model_name.startswith("Qwen") else None
        first_id = lambda char: tokenizer(char, add_special_tokens=False)["input_ids"][0]
        for condition, items in texts.items():
            for slot, (new_pos, copy_pos) in SLOTS.items():
                values = []
                for text in items:
                    text = converter.convert(text) if converter else text
                    new, copied = text[new_pos - 1], text[copy_pos - 1]
                    assert text[slot - 1] == new
                    ids = tokenizer(text[: slot - 1], return_tensors="pt", add_special_tokens=False)["input_ids"]
                    with torch.no_grad():
                        log_probs = torch.log_softmax(model(ids.to(device)).logits[0, -1].float(), dim=-1)
                    top = int(log_probs.argmax())
                    values.append((-log_probs[first_id(new)].item(), -log_probs[first_id(copied)].item(),
                                   top == first_id(new), top == first_id(copied)))
                v = np.array(values, dtype=float)
                rows.append({"model": model_name.split("/")[-1].removesuffix("-Base"), "condition": condition,
                             "slot": slot, "n": len(values),
                             "surprisal_new": v[:, 0].mean(), "surprisal_copied": v[:, 1].mean(),
                             "top_new": v[:, 2].mean(), "top_copied": v[:, 3].mean()})
                print(rows[-1])
        del model

    output = Path(args.output)
    write_json(output, {"slots": {str(k): v for k, v in SLOTS.items()}, "rows": rows})
    lines = ["| Model | Condition | Slot | Surprisal, new item | Surprisal, copied item | Top = new | Top = copied |",
             "|:--|:--|:--:|--:|--:|--:|--:|"]
    lines += [f"| {r['model']} | {r['condition']} | {r['slot']} | {r['surprisal_new']:.2f} | "
              f"{r['surprisal_copied']:.2f} | {r['top_new']:.0%} | {r['top_copied']:.0%} |" for r in rows]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {output} and {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
