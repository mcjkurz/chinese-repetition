"""Collect the essays of 欽定四書文 (方苞, 1738) and punctuate them automatically.

The anthology is read from Wikisource (欽定四書文 (四庫全書本), volumes 化治, 正嘉,
隆萬, 啟禎 and 本朝), where it is transcribed from the 四庫全書 without
punctuation. In each volume an essay is a heading line (title, scope, author)
followed by the essay on one line and the editor's comments; the essays are
told apart from the table of contents and the comments by that layout.

Punctuation is added by a classical-Chinese punctuation model
(raynardj/classical-chinese-punctuation-guwen-biaodian) in windows of
--window characters; only the marks that end a fragment are kept. Against six
human-punctuated essays the model finds about 94 percent of the breaks, and
about 88 percent of its breaks are human ones (see the report).

Writes data/4_eight_legged_essays/qinding_sishuwen.json: {"essays": [{"period", "volume", "title",
"author", "text", "punctuated"}]}.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from common import select_device

PUNCTUATION_MODEL = "raynardj/classical-chinese-punctuation-guwen-biaodian"
BREAKS = set("，。；：？！、")
HAN = re.compile(r"[㐀-鿿豈-﫿\U00020000-\U0002ffff]")
VOLUMES = ([f"化治卷{i}" for i in range(1, 7)] + [f"正嘉卷{i}" for i in range(1, 7)]
           + [f"隆萬卷{i}" for i in range(1, 7)] + [f"啟禎卷{i}" for i in range(1, 10)]
           + [f"本朝卷{i:02d}" for i in range(1, 15)])


class Punctuator:
    def __init__(self, device, window: int = 200, batch_size: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(PUNCTUATION_MODEL)
        self.model = AutoModelForTokenClassification.from_pretrained(PUNCTUATION_MODEL).to(device).eval()
        self.device, self.window, self.batch_size = device, window, batch_size

    def __call__(self, text: str) -> str:
        """The text with a fragment-ending mark after every character the model punctuates."""
        chunks = [text[start:start + self.window] for start in range(0, len(text), self.window)]
        out = []
        for index in range(0, len(chunks), self.batch_size):
            batch = chunks[index:index + self.batch_size]
            encoded = self.tokenizer([list(c) for c in batch], is_split_into_words=True, return_tensors="pt",
                                     padding=True).to(self.device)
            with torch.no_grad():
                predicted = self.model(**encoded).logits.argmax(dim=-1).cpu().tolist()
            for row, chunk in enumerate(batch):
                labels = {}
                for position, word in enumerate(encoded.word_ids(row)):
                    if word is not None and word not in labels:
                        labels[word] = self.model.config.id2label[predicted[row][position]]
                for i, char in enumerate(chunk):
                    out.append(char)
                    if labels.get(i) in BREAKS:
                        out.append(labels[i])
        return "".join(out)


def strip_templates(line: str) -> str:
    previous = None
    while previous != line:
        previous, line = line, re.sub(r"\{\{[^{}]*\}\}", "", line)
    return line


def fetch(volume: str) -> str:
    title = f"欽定四書文 (四庫全書本)/{volume}"
    url = f"https://zh.wikisource.org/w/index.php?title={urllib.parse.quote(title)}&action=raw"
    request = urllib.request.Request(url, headers={"User-Agent": "chinese-repetition research script"})
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8")


def essays_in(volume: str, wikitext: str) -> list[dict]:
    body = wikitext.split("<poem>", 1)[-1].split("</poem>")[0]
    lines = [line for line in body.split("\n") if line.strip()]
    essays = []
    for heading, following in zip(lines, lines[1:]):
        if "SK anchor" in heading:  # table of contents
            continue
        head = strip_templates(heading).strip("　 ")
        text = "".join(HAN.findall(strip_templates(following)))
        if len(head) <= 40 and "　" in head and len(text) > 150:
            parts = [p for p in re.split(r"　+", head) if p]
            essays.append({"period": volume[:2], "volume": volume, "title": parts[0],
                           "author": parts[-1].replace(" ", ""), "text": text})
    return essays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, help="Folder with downloaded volumes (<volume>.txt); else fetch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("data/4_eight_legged_essays/qinding_sishuwen.json"))
    args = parser.parse_args()

    essays = []
    for volume in VOLUMES:
        if args.raw_dir and (args.raw_dir / f"{volume}.txt").exists():
            wikitext = (args.raw_dir / f"{volume}.txt").read_text(encoding="utf-8")
        else:
            wikitext = fetch(volume)
            time.sleep(0.5)
        found = essays_in(volume, wikitext)
        essays.extend(found)
        print(f"{volume}: {len(found)} essays")

    punctuate = Punctuator(select_device(args.device))
    for essay in essays:
        essay["punctuated"] = punctuate(essay["text"])
    args.output.write_text(json.dumps({"source": "欽定四書文 (四庫全書本), zh.wikisource.org", "punctuation_model": PUNCTUATION_MODEL,
                                       "essays": essays}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Saved: {args.output} ({len(essays)} essays)")


if __name__ == "__main__":
    main()
