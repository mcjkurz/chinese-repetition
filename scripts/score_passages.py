"""Score next-token surprisal and entropy for every passage in a control set.

Reads a JSON file with a "passages" list (as written by make_sensical_controls.py
or make_nonsense_controls.py), loads the model once, and writes one result
file with per-position surprisal for each passage.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import select_device
from lm_scoring_utils import DEFAULT_MODEL, causal_lm_context_limit
from sequence_surprisal import score_surprisal, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    device = select_device(args.device)
    print(f"Loading model: {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()
    max_context_tokens = causal_lm_context_limit(model, tokenizer)

    passages = []
    for passage in source["passages"]:
        points = score_surprisal(
            model=model,
            tokenizer=tokenizer,
            text=passage["text"],
            device=device,
            max_context_tokens=max_context_tokens,
        )
        if len(points) != len(passage["text"]):
            raise ValueError(f"{passage['id']}: expected one token per character.")
        passages.append(
            {
                **passage,
                "mean_surprisal_nats": float(
                    np.mean([p["surprisal_nats"] for p in points if p["surprisal_nats"] is not None])
                ),
                "mean_entropy_nats": float(
                    np.mean([p["entropy_nats"] for p in points if p["entropy_nats"] is not None])
                ),
                "points": points,
            }
        )
        print(f"{passage['id']}: mean {passages[-1]['mean_surprisal_nats']:.2f} nats")

    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": args.input,
            "model": args.model,
            "device": str(device),
            "metric": "next-token surprisal in nats: -ln p(x_t | x_<t)",
            "entropy": "entropy in nats of the next-token distribution p(. | x_<t)",
            "first_token": "unscored because no preceding token is provided",
        },
        "passages": passages,
    }
    write_json(Path(args.output), payload)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
