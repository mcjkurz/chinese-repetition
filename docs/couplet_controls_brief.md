# Control couplets: brief for classifiers and writers

Two kinds of control couplets are built on 1,000 inner (parallel) couplets of
each meter, sampled from the held-out test set
(`data/2-3_regulated_verse/couplet_controls/{penta,hepta}_inner.json`), for the decoding and
encoding experiments on regulated verse. Everything is checked and built by
`scripts/make_couplet_controls.py`. Text is in simplified characters, as in
the couplet corpus. Work from the project root with `venv/bin/python`, keep
any helper files in your own private folder, and never write to another
worker's file.

## Part 1: word classes (for the nonsense couplets)

Nonsense couplets replace every character with a random character of the same
word class, so that the parallelism of the two lines survives while their
sense is lost. You classify one list of characters,
`data/2-3_regulated_verse/couplet_controls/lexicon/chars_<k>.txt` (character, corpus frequency,
two example lines), and write `data/2-3_regulated_verse/couplet_controls/lexicon/classes_<k>.tsv`,
one line per character: the character, a tab, and one class tag.

| Tag | Class | Examples |
|---|---|---|
| N | noun: things, beings, places, times, seasons, names | 山 風 月 人 君 春 夜 秋 家 |
| V | verb: actions and processes | 看 聞 落 開 歸 送 照 流 |
| A | adjective or stative verb (not colour) | 高 深 寒 清 長 孤 新 老 |
| C | colour | 白 青 紅 綠 碧 黃 翠 素 |
| M | numeral or quantity | 一 三 萬 千 百 半 兩 數 |
| D | direction or position | 東 西 南 北 上 下 中 間 前 後 外 |
| P | pronoun | 我 吾 爾 誰 此 其 彼 |
| F | function word: adverb, preposition, conjunction, particle | 不 無 未 已 猶 自 相 欲 將 又 於 以 與 而 之 |

- Give the class of the character's dominant use in regulated verse, judged
  from your knowledge and the two example lines; one class per character.
- A character that is mainly a noun but often used as a verb (e.g. 雨, 衣)
  keeps the class of its most frequent use.
- Colours, numerals and directions have their own classes because classical
  parallelism pairs them with each other (顏色對, 數目對, 方位對); use them
  even where a colour or numeral word acts as an adjective.
- Classify every character of your list; do not skip any.
- Check with `venv/bin/python scripts/make_couplet_controls.py check-lexicon`
  (it reports problems in all files; fix only yours).

## Part 2: composed couplets

A composed couplet is a new parallel couplet written after one real inner
couplet. You receive `data/2-3_regulated_verse/couplet_controls/composed/sources_<k>.txt` (id, tab,
source couplet) and write `data/2-3_regulated_verse/couplet_controls/composed/composed_<k>.txt`,
one line per source: the same id, a tab, and your couplet as `line1，line2`.

- Same number of characters per line as the source (5 or 7).
- Strictly parallel, as an inner couplet: each position of the second line
  answers the same position of the first in word class (noun for noun, verb
  for verb, colour for colour, numeral for numeral), and the two lines are
  built alike.
- Write about another subject than the source: other images, other
  situation. Follow the source's rhythm and syntax where it comes naturally,
  but do not copy it word for word: no run of three characters from the
  source couplet.
- The couplet must be sound classical Chinese that could stand in a regulated
  poem; tonal rules are not required. Avoid *hezhang* 合掌: the second line
  must not merely restate the first.
- Do not quote: no line may occur verbatim in the corpus (the check finds
  this), and do not reproduce couplets you know.
- Use common characters only: every character must be among the 3,000 most
  frequent of the corpus (the check reports any that are not).
- Do not reuse one couplet frame mechanically; vary subjects, images and
  constructions across your couplets.
- Check with `venv/bin/python scripts/make_couplet_controls.py check-composed
  data/2-3_regulated_verse/couplet_controls/composed/composed_<k>.txt` until it reports no
  problems. Save after every twenty couplets or so.

## Report

When done, report the number of characters classified or couplets written,
the check result, and any hard cases.
