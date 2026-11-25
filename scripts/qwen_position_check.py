"""Check whether a Qwen3 base model copies or completes the analogy at character 57.

At character 57 (生 in the Mengzi), the passage repeats the clause 舍 X 而取 Y with
the new items. A model that copies the preceding string predicts the item at
character 25 (魚); a model that completes the analogy predicts the item at
character 57 itself. Texts are converted to simplified characters first, since
Qwen splits probability between traditional and simplified forms.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import opencc
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import select_device
from sequence_surprisal import write_json


DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
MENGZI = "data/1_mengzi_prose/mengzi_surprisal.txt"
CONTROLS = {
    "composed": "data/1_mengzi_prose/surprisal_controls/sensical.json",
    "nonsense": "data/1_mengzi_prose/surprisal_controls/nonsense.json",
}
COPY_CHAR = 25
TARGET_CHAR = 57


def load_passages() -> list[dict[str, str]]:
    passages = [{"id": "mengzi", "condition": "mengzi", "text": Path(MENGZI).read_text(encoding="utf-8").strip()}]
    for condition, path in CONTROLS.items():
        for passage in json.loads(Path(path).read_text(encoding="utf-8"))["passages"]:
            passages.append({"id": passage["id"], "condition": condition, "text": passage["text"]})
    return passages


def continuation_log_prob(model, tokenizer, prefix_ids: list[int], continuation: str, device) -> float:
    """Return ln p(continuation | prefix), summed over the continuation's tokens."""
    continuation_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([prefix_ids + continuation_ids], device=device)
    with torch.no_grad():
        log_probs = torch.log_softmax(model(input_ids).logits[0].float(), dim=-1)
    start = len(prefix_ids) - 1
    return sum(log_probs[start + i, token].item() for i, token in enumerate(continuation_ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="results/surprisal/qwen3_0.6b_base_position57.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    converter = opencc.OpenCC("t2s")
    device = select_device(args.device)
    print(f"Loading model: {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.to(device)
    model.eval()

    results = []
    for passage in tqdm(load_passages(), desc="Passages"):
        traditional = passage["text"]
        text = converter.convert(traditional)
        if len(text) != len(traditional):
            raise ValueError(f"{passage['id']}: conversion changed the length.")
        prefix = text[: TARGET_CHAR - 1]
        copy_item, target_item = text[COPY_CHAR - 1], text[TARGET_CHAR - 1]

        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            row = torch.log_softmax(
                model(torch.tensor([prefix_ids], device=device)).logits[0, -1].float(), dim=-1
            )
        top = torch.topk(row, args.top_k)
        predictions = [
            {"token": tokenizer.decode([i]), "surprisal_nats": -v}
            for v, i in zip(top.values.tolist(), top.indices.tolist())
        ]
        top_token = predictions[0]["token"]
        if copy_item == target_item:
            outcome = "ambiguous"
        elif top_token.startswith(target_item):
            outcome = "correct"
        elif top_token.startswith(copy_item):
            outcome = "copy"
        else:
            outcome = "other"

        results.append(
            {
                **passage,
                "text_simplified": text,
                "copy_item": copy_item,
                "target_item": target_item,
                "target_surprisal_nats": -continuation_log_prob(model, tokenizer, prefix_ids, target_item, device),
                "copy_surprisal_nats": -continuation_log_prob(model, tokenizer, prefix_ids, copy_item, device),
                "top_predictions": predictions,
                "outcome": outcome,
            }
        )

    summary = {}
    for condition in ["mengzi", *CONTROLS]:
        rows = [r for r in results if r["condition"] == condition]
        summary[condition] = {"n": len(rows), **Counter(r["outcome"] for r in rows)}
        print(f"{condition}: {summary[condition]}")

    write_json(
        Path(args.output),
        {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "device": str(device),
                "conversion": "OpenCC t2s",
                "copy_char": COPY_CHAR,
                "target_char": TARGET_CHAR,
                "outcome": "top prediction at target_char: correct (target item), copy (item at copy_char), other",
            },
            "summary": summary,
            "passages": results,
        },
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
