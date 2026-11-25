"""Next-character entropy and surprisal of a GujiGPT fine-tuned on degraded or enciphered couplets.

Scores one model of run_degraded_training.sh on held-out couplets, each under
the inner (I) and the outer (O) prompt, as score_couplet_controls.py scores
the model fine-tuned on real couplets:

- recombined, wordswap, nonsense, shuffled (make_degraded_training.py): the
  held-out set of the condition (segswap, wordswap, pairswap, charshuf in
  data/2-3_regulated_verse/couplet_controls) and the real inner and outer couplets;
- charcipher, wordcipher (make_cipher_couplets.py): the enciphered held-out
  inner and outer couplets (data/2-3_regulated_verse/cipher), taken as already in traditional
  characters.

For the conditions whose held-out sets score_couplet_controls.py does not
score (wordswap and the ciphers), the same sets are also scored by the model
fine-tuned on real couplets (I and O prompts) and by the untuned GujiGPT, as
references (keys "ref_finetuned/..." and "ref_base/none/..."). Values are kept
per couplet and position.

Writes --output (default results/degraded_training/decoding_{meter}_{condition}.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import select_device
from decoding_io import DEFAULT_BASE_MODEL_PATH
from score_couplet_controls import load, score_set

TEST_SETS = {"recombined": "segswap", "wordswap": "wordswap", "nonsense": "pairswap", "shuffled": "charshuf",
             "charcipher": "cipher_inner", "wordcipher": "cipher_inner"}
CIPHERS = {"charcipher", "wordcipher"}
NEEDS_REFERENCES = {"wordswap", "charcipher", "wordcipher"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meter", choices=["penta", "hepta"], required=True)
    parser.add_argument("--condition", choices=list(TEST_SETS), required=True)
    parser.add_argument("--model", required=True, help="Path of the fine-tuned model")
    parser.add_argument("--data-dir", default="data/2-3_regulated_verse/couplet_controls")
    parser.add_argument("--cipher-dir", default="data/2-3_regulated_verse/cipher")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N couplets per set (smoke tests)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.condition in CIPHERS:
        paths = {f"cipher_{kind}": Path(args.cipher_dir) / f"{args.meter}_{args.condition}_{kind}.json"
                 for kind in ("inner", "outer")}
    else:
        paths = {name: Path(args.data_dir) / f"{args.meter}_{name}.json"
                 for name in (TEST_SETS[args.condition], "inner", "outer")}
    sets = {}
    for name, path in paths.items():
        items = json.loads(path.read_text(encoding="utf-8"))
        sets[name] = items[:args.limit] if args.limit else items
    convert = args.condition not in CIPHERS

    device = select_device()
    result = {"meter": args.meter, "condition": args.condition, "model": args.model,
              "own_set": TEST_SETS[args.condition], "scores": {}}

    def run(model, tokenizer, prefix: str, prompts: dict[str, str], names) -> None:
        for prompt_name, prompt in prompts.items():
            for name in names:
                key = f"{prefix}{prompt_name}/{name}"
                result["scores"][key] = score_set(sets[name], model, tokenizer, prompt, device, args.batch_size,
                                                  convert=convert)
                entropy = np.array([r["entropy"] for r in result["scores"][key]]).mean(axis=0)
                loss = np.mean([np.mean(r["loss"]) for r in result["scores"][key]])
                print(f"{args.meter} {args.condition} {key}: entropy by position "
                      f"{' '.join(f'{e:.2f}' for e in entropy)}, mean loss {loss:.3f}", flush=True)

    model, tokenizer = load(args.model, device)
    run(model, tokenizer, "", {"I": "i", "O": "o"}, list(sets))
    del model
    if args.condition in NEEDS_REFERENCES:
        own = [n for n in sets if n not in ("inner", "outer")]
        model, tokenizer = load(f"models/gujigpt-io-finetuned-{args.meter}", device)
        run(model, tokenizer, "ref_finetuned/", {"I": "i", "O": "o"}, own)
        del model
        model, tokenizer = load(DEFAULT_BASE_MODEL_PATH, device)
        run(model, tokenizer, "ref_base/", {"none": ""}, own)
        del model

    output = Path(args.output or f"results/degraded_training/decoding_{args.meter}_{args.condition}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
