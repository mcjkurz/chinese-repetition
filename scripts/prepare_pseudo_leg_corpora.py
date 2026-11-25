"""Prepare non-parallel classical prose for the pseudo-leg comparison.

Two corpora, both written as {"source", "pieces": [{"title", "text"}]} in
traditional characters:

- 古文觀止, volumes 8-12 (Tang, Song and Ming prose), from the raw wikitext of
  zh.wikisource.org/wiki/古文觀止/卷N; the commentary notes ({{*|...}}) and
  other markup are removed, and rhapsodies (賦) are left out. Volume 7, which
  holds most of the parallel prose (駢文), is not used.
- 明史, the biographies (列傳), from the JSON files of
  github.com/quzhi1/ChineseHistoricalSource (simplified characters, converted
  with OpenCC s2t); each paragraph is a piece.

The raw downloads are passed as arguments and not kept in the project.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import opencc

SECTION = re.compile(r"^==\s*([^=]+?)\s*==\s*$(.*?)(?=^==[^=]|\Z)", re.M | re.S)


def strip_wikitext(text: str) -> str:
    previous = None
    while previous != text:  # templates can nest
        previous, text = text, re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"'''?|-\{|\}-", "", text)
    return re.sub(r"\s+", "", text)


def guwen_pieces(paths: list[Path]) -> list[dict]:
    pieces = []
    for path in paths:
        for match in SECTION.finditer(path.read_text(encoding="utf-8")):
            title = match.group(1).strip()
            if "賦" not in title:
                pieces.append({"title": f"{path.stem}/{title}", "text": strip_wikitext(match.group(2))})
    return pieces


def mingshi_pieces(path: Path) -> list[dict]:
    converter = opencc.OpenCC("s2t")
    return [{"title": converter.convert(item["chapter"]), "text": converter.convert(item["text"])}
            for item in json.loads(path.read_text(encoding="utf-8")) if "列传" in item["chapter"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guwen", nargs="+", type=Path, required=True,
                        help="Raw wikitext of 古文觀止 volumes, named juan8.txt ... juan12.txt")
    parser.add_argument("--mingshi", type=Path, required=True, help="明史.json from ChineseHistoricalSource")
    parser.add_argument("--output-dir", type=Path, default=Path("data/4_eight_legged_essays/pseudo_legs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, source, pieces in (("guwen", "古文觀止 卷8-12", guwen_pieces(args.guwen)),
                                 ("mingshi", "明史 列傳", mingshi_pieces(args.mingshi))):
        output = args.output_dir / f"{name}.json"
        output.write_text(json.dumps({"source": source, "pieces": pieces}, ensure_ascii=False, indent=1) + "\n",
                          encoding="utf-8")
        print(f"{output}: {len(pieces)} pieces, {sum(len(p['text']) for p in pieces)} characters")


if __name__ == "__main__":
    main()
