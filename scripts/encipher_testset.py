"""Encipher the held-out test set with an existing character cipher.

Reads data/2-3_regulated_verse/couplets_{meter}_test.json (simplified), converts each couplet to
traditional characters with OpenCC as finetune_io_gpt.py does, applies the
character mapping of make_cipher_couplets.py (data/2-3_regulated_verse/cipher/{meter}_charcipher_map.json,
the mapping with which the models of models/cipher_finetune were trained) and
writes data/2-3_regulated_verse/cipher/{meter}_charcipher_test.json with every field kept,
"couplet" enciphered (traditional) and "source_couplet" the traditional
original. Characters absent from the mapping are kept and counted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import opencc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meters", nargs="+", default=["penta", "hepta"])
    parser.add_argument("--cipher-dir", default="data/2-3_regulated_verse/cipher")
    args = parser.parse_args()

    to_traditional = opencc.OpenCC("s2t")
    for meter in args.meters:
        mapping = json.loads((Path(args.cipher_dir) / f"{meter}_charcipher_map.json").read_text(encoding="utf-8"))["mapping"]
        test = json.loads(Path(f"data/2-3_regulated_verse/couplets_{meter}_test.json").read_text(encoding="utf-8"))
        unmapped, total, out = Counter(), 0, []
        for item in test:
            trad = to_traditional.convert(item["couplet"])
            for c in trad:
                if c != "，":
                    total += 1
                    if c not in mapping:
                        unmapped[c] += 1
            out.append({**item, "couplet": "".join(mapping.get(c, c) for c in trad), "source_couplet": trad})
        path = Path(args.cipher_dir) / f"{meter}_charcipher_test.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{meter}: {len(out)} couplets, labels {dict(Counter(o['label'] for o in out))}; "
              f"{sum(unmapped.values())} of {total} character tokens unmapped ({len(unmapped)} types): {path}")


if __name__ == "__main__":
    main()
