"""Transition alignment of the control couplets, per couplet and layer.

Encodes every couplet of the sets of make_couplet_controls.py (real inner,
real outer, composed and nonsense couplets of each meter) as
transition_analysis.py does: the couplet in traditional characters with its
punctuation ("line1，line2。") is passed through GujiBERT, and at every layer
the alignment is the mean cosine between corresponding character-to-character
transitions of the two lines. The position-matched random-pair baseline of the
main experiment (results/transition_gujibert_paper, estimated from 100,000
real couplets of the meter) is subtracted, so that the values are on the scale
of the main figure. Sets not built yet are skipped.

Writes results/couplet_controls/alignment_<meter>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from common import select_device
from transition_analysis import (DEFAULT_MODEL, couplet_context_and_targets, get_character_embeddings_batch,
                                 layer_transition_alignment, token_count_matches_text_length)

SETS = ("inner", "outer", "composed", "nonsense", "pairswap", "segswap", "charshuf")
BASELINE = "results/transition_gujibert_paper/transition_alignment_random_pair_baseline_{meter}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meters", nargs="+", default=["penta", "hepta"])
    parser.add_argument("--data-dir", default="data/2-3_regulated_verse/couplet_controls")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="results/couplet_controls")
    args = parser.parse_args()

    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModel.from_pretrained(DEFAULT_MODEL, output_hidden_states=True).to(device).eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for meter in args.meters:
        baseline = np.array(json.loads(Path(BASELINE.format(meter=meter)).read_text())["random_pair_baseline"])
        result = {"meter": meter, "baseline": baseline.tolist(), "alignment": {}}
        for name in SETS:
            path = Path(args.data_dir) / f"{meter}_{name}.json"
            if not path.exists():
                continue
            items = json.loads(path.read_text(encoding="utf-8"))
            records, skipped = [], 0
            for start in tqdm(range(0, len(items), args.batch_size), desc=f"{meter} {name}", leave=False):
                batch = []
                for item in items[start:start + args.batch_size]:
                    text, targets = couplet_context_and_targets(item["line1_traditional"], item["line2_traditional"], True)
                    if token_count_matches_text_length(text, tokenizer):
                        batch.append((item, text, targets))
                    else:
                        skipped += 1
                if not batch:
                    continue
                embeddings = get_character_embeddings_batch([t for _, t, _ in batch], [g for _, _, g in batch],
                                                            tokenizer, model, device)
                for (item, _, _), layers in zip(batch, embeddings):
                    n = len(item["line1_traditional"])
                    raw = layer_transition_alignment(layers[:, :n], layers[:, n:])["layer_mean_alignments"]
                    records.append({"id": item["id"], "alignment": np.round(np.array(raw) - baseline, 4).tolist()})
            result["alignment"][name] = records
            values = np.array([r["alignment"] for r in records])
            print(f"{meter} {name}: {len(records)} couplets ({skipped} skipped), layer 8 minus baseline "
                  f"{values[:, 8].mean():.3f}")
        path = output_dir / f"alignment_{meter}.json"
        path.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
