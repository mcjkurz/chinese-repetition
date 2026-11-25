"""Next-character entropy and surprisal of the second line, per couplet, for the control couplets.

Scores the sets of make_couplet_controls.py (real inner, real outer, composed
and nonsense couplets of each meter) as decoding_io.py scores the test set,
with its get_next_token_scores: every character of the second line is
predicted from the first line, the comma and the true preceding characters
(teacher forcing), and the entropy of the next-character distribution and the
surprisal of the actual character are recorded. Each couplet is scored by the
fine-tuned GujiGPT of its meter under the inner (I) and the outer (O) prompt,
and by the untuned GujiGPT without a prompt. Unlike decoding_io.py, the values
are kept per couplet and position, so that the sets can be compared couplet by
couplet. Sets not built yet are skipped.

Writes results/couplet_controls/decoding_<meter>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import opencc
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ensure_padding_token, select_device, split_couplet
from decoding_io import DEFAULT_BASE_MODEL_PATH, get_next_token_scores

SETS = ("inner", "outer", "composed", "nonsense", "pairswap", "segswap", "charshuf")
MODELS = {"finetuned": "models/gujigpt-io-finetuned-{meter}", "base": DEFAULT_BASE_MODEL_PATH}


def load(path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(path)
    ensure_padding_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(path)
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.pad_token_id or tokenizer.pad_token_id
    return model.to(device).eval(), tokenizer


def score_set(items: list[dict], model, tokenizer, prompt: str, device, batch_size: int,
              convert: bool = True) -> list[dict]:
    """Entropy and surprisal at every position of the second line, per couplet.

    With convert=False the couplets are taken as already in traditional characters."""
    converter = opencc.OpenCC("s2t") if convert else None
    out = []
    for start in tqdm(range(0, len(items), batch_size), leave=False):
        batch = items[start:start + batch_size]
        prefixes, targets, owners = [], [], []
        for index, item in enumerate(batch):
            first, second = split_couplet(converter.convert(item["couplet"]) if converter else item["couplet"])
            for k in range(len(second)):
                prefixes.append(prompt + first + "，" + second[:k])
                targets.append(second[k])
                owners.append(index)
        scores = get_next_token_scores(model, tokenizer, prefixes, targets, device)
        for index, item in enumerate(batch):
            mine = [s for s, o in zip(scores, owners) if o == index]
            out.append({"id": item["id"], "entropy": [round(s["entropy"], 4) for s in mine],
                        "loss": [round(s["loss"], 4) for s in mine]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meters", nargs="+", default=["penta", "hepta"])
    parser.add_argument("--data-dir", default="data/2-3_regulated_verse/couplet_controls")
    parser.add_argument("--batch-size", type=int, default=16, help="Couplets per batch")
    parser.add_argument("--output-dir", default="results/couplet_controls")
    args = parser.parse_args()

    device = select_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for meter in args.meters:
        sets = {name: json.loads(path.read_text(encoding="utf-8")) for name in SETS
                if (path := Path(args.data_dir) / f"{meter}_{name}.json").exists()}
        result = {"meter": meter, "sets": sorted(sets), "scores": {}}
        for model_name, template in MODELS.items():
            model, tokenizer = load(template.format(meter=meter), device)
            prompts = {"I": "i", "O": "o"} if model_name == "finetuned" else {"none": ""}
            for prompt_name, prompt in prompts.items():
                for name, items in sets.items():
                    key = f"{model_name}/{prompt_name}/{name}"
                    result["scores"][key] = score_set(items, model, tokenizer, prompt, device, args.batch_size)
                    entropy = np.mean([np.mean(r["entropy"]) for r in result["scores"][key]])
                    loss = np.mean([np.mean(r["loss"]) for r in result["scores"][key]])
                    print(f"{meter} {key}: entropy {entropy:.3f}, loss {loss:.3f}")
            del model
        path = output_dir / f"decoding_{meter}.json"
        path.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
