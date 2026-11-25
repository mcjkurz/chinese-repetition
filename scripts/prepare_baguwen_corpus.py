"""Collect eight-legged essays from Wikisource and mark their parallel leg pairs.

The essays are the members of zh.wikisource.org Category:八股文 other than Chen
Zilong's (which is data/4_eight_legged_essays/baguwen_chen_zilong.txt). Each is downloaded as raw
wikitext and cleaned of templates and line breaks. Their legs are not marked
in the source, so the pairs of parallel passages are given below by the words
that open and close each member; a passage runs from its opening words to the
first occurrence of its closing words after them, so that every leg is cut
from the downloaded text and not retyped. A pair is kept only if its two
members have the same number of punctuation-delimited fragments.

Writes data/4_eight_legged_essays/baguwen_corpus.json: {"essays": [{"id", "title", "author", "date",
"url", "text", "pairs": [{"a", "b", "fragments"}]}]}.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

FRAGMENT_SPLIT = re.compile(r"[，。；：？！、?!]")

# (opening and closing words of member A, opening and closing words of member B)
ESSAYS = [
    {"id": "fei_hong", "author": "費宏", "date": "Ming",
     "title": "君子不重則不威，學而不固，主忠信，無友不如己者，過則勿憚改",
     "pairs": [("若言動之間", "既不足於厚重；", "則應酬之際", "亦不見其威嚴。"),
               ("雖曰學以明善", "未必服膺而勿失也；", "雖曰學以復初", "未必力行以求至也。"),
               ("惟誠乃善之基也", "不矯偽而無物焉；", "惟偽乃惡之門也", "不欺詐而無實焉。"),
               ("取友所以輔仁", "豈可泥交而苟合乎?", "改過乃能入德", "豈可畏難而苟安乎?"),
               ("內外相須", "而其功不可缺；", "終始相因", "而其序不可亂，")]},
    {"id": "gui_youguang", "author": "歸有光", "date": "Ming",
     "title": "大學之道在明明德在親民在止於至善",
     "pairs": [("己之德所當明也", "道不盡於己，非所以為學矣。", "民之德所當新也", "道不盡於人，非所以為學矣。"),
               ("有憲天之學", "可以言格天之功；", "有格天之功", "可以言配天之治。"),
               ("不與天地合", "猶為未明之德也；", "不與三代同", "猶為未新之民也。")]},
    {"id": "han_tan", "author": "韓菼", "date": "1673",
     "title": "子謂顏淵曰，用之則行，捨之則藏，惟我與爾有是夫",
     "pairs": [("汲於行者蹶", "非復功名中人也。", "一於藏者緩", "非復泉石中人也。"),
               ("則嘗試擬而求之", "此際亦差堪慰耳。", "則又嘗身爲試之", "此際亦足共慰耳。"),
               ("惟我與爾攬事物之歸", "共千古已矣。", "惟我與爾參神明之變", "予造物已矣。")]},
    {"id": "wang_shouren", "author": "王守仁", "date": "Ming",
     "title": "志士仁人，無求生以害仁，有殺生以成仁",
     "pairs": [("夫所謂志士者", "有植天下之大閒；", "所謂仁人者", "以貞天下之大節。"),
               ("是以其禍患之方殷", "而彼有所不屑也。", "變故之偶值", "而彼有所不為也。"),
               ("理欲無並立之機", "雖至死而靡憾。", "心跡無兩全之勢", "雖滅身而無悔。"),
               ("當國家傾覆之徐", "而存亡由之不計矣。", "值顛沛流離之餘", "而死生由之勿恤矣。"),
               ("以吾心為重", "誠何如哉？", "以存心為生", "又何如哉？"),
               ("是知觀志士之所為", "無志者可以愧矣，", "觀仁人之所為", "不仁者可以思矣。")]},
    {"id": "wang_ao", "author": "王鏊", "date": "1475",
     "title": "百姓足，君孰與不足",
     "pairs": [("民力所出", "不困於徵求；", "民財所有", "不盡於聚斂。"),
               ("間閻之內", "仰事俯育者無憂矣；", "田野之間", "養生送死者無憾矣。"),
               ("吾知藏諸閭閻者", "而後爲吾財也；", "蓄諸田野者", "而後爲吾有也。"),
               ("取之無窮", "有求而不得？", "用之不竭", "有事而無備？"),
               ("犧牲粢盛", "其孰與不足乎？", "饔饗牢醴", "又孰與不足乎？")]},
]


def fetch(title: str) -> str:
    url = f"https://zh.wikisource.org/w/index.php?title={urllib.parse.quote(title)}&action=raw"
    request = urllib.request.Request(url, headers={"User-Agent": "chinese-repetition research script"})
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8")


def clean(wikitext: str) -> str:
    text = re.sub(r"\{\{header.*?\n\}\}", "", wikitext, flags=re.S)
    previous = None
    while previous != text:  # templates can nest
        previous, text = text, re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"<br\s*/?>", "", text)
    return re.sub(r"\s+", "", text)


def passage(text: str, opening: str, closing: str) -> str:
    start = text.index(opening)
    return text[start:text.index(closing, start) + len(closing)]


def fragments(text: str) -> list[str]:
    return [f for f in FRAGMENT_SPLIT.split(text) if f]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/4_eight_legged_essays/baguwen_corpus.json"))
    args = parser.parse_args()

    essays = []
    for spec in ESSAYS:
        text = clean(fetch(spec["title"]))
        pairs = []
        for open_a, close_a, open_b, close_b in spec["pairs"]:
            a, b = passage(text, open_a, close_a), passage(text, open_b, close_b)
            if len(fragments(a)) != len(fragments(b)):
                print(f"{spec['id']}: skipped a pair with {len(fragments(a))} and {len(fragments(b))} fragments")
                continue
            pairs.append({"a": a, "b": b, "fragments": len(fragments(a))})
        essays.append({"id": spec["id"], "title": spec["title"], "author": spec["author"], "date": spec["date"],
                       "url": f"https://zh.wikisource.org/wiki/{urllib.parse.quote(spec['title'])}",
                       "text": text, "pairs": pairs})
        print(f"{spec['id']}: {len(pairs)} pairs, fragments per member {[p['fragments'] for p in pairs]}")

    args.output.write_text(json.dumps({"source": "zh.wikisource.org Category:八股文", "essays": essays},
                                      ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
