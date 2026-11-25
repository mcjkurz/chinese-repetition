"""Layer-wise paired-leg self-similarity for an essay and its control essays.

For the original essay and every essay in each control set, compute centered
GujiBERT self-similarity matrices at every layer, the paired-leg metrics of
self_similarity_matrix.leg_similarity_metrics, and the same metrics for
character-scrambled versions of that essay. Writes per-essay results and a
per-condition, per-layer summary.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from transformers import AutoModel, AutoTokenizer

from common import select_device
from self_similarity_matrix import (
    DEFAULT_MODEL,
    all_layer_character_embeddings,
    cosine_similarity_matrix,
    leg_similarity_metrics,
    parse_leg_marked_text,
    scramble_text,
)

METRICS = {
    "pair_contrast": ("paired_leg_units", "within_minus_between"),
    "paired_cross_contrast": ("paired_vs_non_paired_legs", "paired_cross_minus_non_paired_cross"),
}


def layer_metrics(text: str, leg_spans, model, tokenizer, device) -> list[dict[str, float]]:
    chars, hidden = all_layer_character_embeddings(text, model, tokenizer, device)
    if len(chars) != len(text):
        raise ValueError(f"Tokenized {len(chars)} visible tokens from {len(text)} characters")
    results = []
    for layer_hidden in hidden:
        metrics = leg_similarity_metrics(cosine_similarity_matrix(layer_hidden, center=True), leg_spans)
        results.append({name: metrics[group][key] for name, (group, key) in METRICS.items()})
    return results


def score_essay(essay_id: str, raw_text: str, baseline_count: int, rng, model, tokenizer, device) -> dict:
    text, leg_spans, _ = parse_leg_marked_text(raw_text)
    observed = layer_metrics(text, leg_spans, model, tokenizer, device)
    baselines = [
        layer_metrics(scramble_text(text, rng), leg_spans, model, tokenizer, device)
        for _ in range(baseline_count)
    ]
    layers = []
    for layer_index, values in enumerate(observed):
        entry = {"layer": layer_index}
        for name in METRICS:
            scrambled = np.array([baseline[layer_index][name] for baseline in baselines])
            entry[name] = values[name]
            entry[f"{name}_scrambled_mean"] = float(scrambled.mean()) if scrambled.size else None
            entry[f"{name}_scrambled_p95"] = float(np.percentile(scrambled, 95)) if scrambled.size else None
            entry[f"{name}_p"] = (
                float((np.sum(scrambled >= values[name]) + 1) / (scrambled.size + 1)) if scrambled.size else None
            )
        layers.append(entry)
    print(f"{essay_id}: " + " ".join(f"{entry['pair_contrast']:.3f}" for entry in layers))
    return {"id": essay_id, "length": len(text), "leg_count": len(leg_spans), "layers": layers}


def summarize(essays: list[dict]) -> list[dict]:
    summary = []
    for layer_index in range(len(essays[0]["layers"])):
        entry = {"layer": layer_index}
        for name in METRICS:
            for suffix in ("", "_scrambled_mean"):
                values = np.array([essay["layers"][layer_index][name + suffix] for essay in essays], dtype=float)
                entry[f"{name}{suffix}_mean"] = float(values.mean())
                entry[f"{name}{suffix}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            p_values = [essay["layers"][layer_index][f"{name}_p"] for essay in essays]
            entry[f"{name}_share_p_below_05"] = float(np.mean([p < 0.05 for p in p_values]))
        summary.append(entry)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="data/4_eight_legged_essays/baguwen_chen_zilong.txt")
    parser.add_argument(
        "--controls",
        nargs="*",
        default=["composed=data/4_eight_legged_essays/baguwen_controls/sensical.json", "nonsense=data/4_eight_legged_essays/baguwen_controls/nonsense.json"],
        help="label=path pairs; each JSON has a 'passages' list of {id, text}",
    )
    parser.add_argument("--original-baselines", type=int, default=1000)
    parser.add_argument("--control-baselines", type=int, default=100)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/baguwen_self_similarity/summary.json")
    args = parser.parse_args()

    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    rng = random.Random(args.seed)

    conditions = {
        "original": [
            score_essay("original", Path(args.original).read_text(encoding="utf-8"),
                        args.original_baselines, rng, model, tokenizer, device)
        ]
    }
    for spec in args.controls:
        label, path = spec.split("=", 1)
        passages = json.loads(Path(path).read_text(encoding="utf-8"))["passages"]
        conditions[label] = [
            score_essay(passage["id"], passage["text"], args.control_baselines, rng, model, tokenizer, device)
            for passage in passages
        ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "original": args.original,
        "controls": args.controls,
        "original_baselines": args.original_baselines,
        "control_baselines": args.control_baselines,
        "seed": args.seed,
        "summary": {label: summarize(essays) for label, essays in conditions.items()},
        "essays": conditions,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
