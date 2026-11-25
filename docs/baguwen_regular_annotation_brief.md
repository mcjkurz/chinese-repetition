# Brief for annotating regular eight-legged essays (欽定四書文)

This brief is given to every annotator, so that the annotation can be repeated
or extended. The method it serves is described in
`docs/baguwen_leg_alignment_method.md`.

## Task

Read every essay in your index range of `data/4_eight_legged_essays/qinding_sishuwen.json` and
decide whether it is a regular eight-legged essay. If it is, mark its four leg
pairs; if it is not, give the reason in a few words.

Work from the project root with the project's Python:

    venv/bin/python scripts/check_regular_legs.py show 120 121 122
    venv/bin/python scripts/check_regular_legs.py check data/4_eight_legged_essays/qinding_regular/part_N.json
    venv/bin/python scripts/check_regular_legs.py check --quiet data/4_eight_legged_essays/qinding_regular/part_N.json

`show` prints essays with their automatic punctuation; `check` cuts your legs
out of the text and prints them fragment by fragment, so that you can see
whether you marked what you meant, and lists problems.

## What counts as regular

An essay is regular when its body has exactly four leg pairs: 起股 (or 提比),
中股 (中比), 後股 (後比) and 束股 (束比), that is eight legs, after the opening
sections (破題, 承題, 起講, 入題/領題) and before the conclusion (收結/大結).

- A pair is two legs that answer each other clause by clause: parallel
  syntax, matching or contrasting terms, usually the same number of clauses.
  The two legs of a pair are adjacent in the text: leg A, then leg B, then the
  next pair. The order in the file must be the text order 1A 1B 2A 2B 3A 3B
  4A 4B.
- A leg may be of any length, from a single clause to a long paragraph. Do
  not skip a pair because it is short; a short 束股 still counts.
- Parallel sentences inside one leg are not separate pairs. If leg A contains
  four parallel sentences that are answered by four sentences in leg B, that
  is one pair, not four.
- Linking sentences between pairs (出題, 過接, e.g. a sentence that turns from
  one pair to the next or restates the topic) belong to no leg; leave them
  out.
- Not regular: three or fewer pairs, five or more pairs, a body written as
  continuous prose, a pair whose two legs do not answer each other, or a
  structure you cannot divide into four pairs with confidence. When in doubt,
  mark the essay as not regular and say why; a smaller clean sample is better
  than a larger doubtful one.

## Format

Write `data/4_eight_legged_essays/qinding_regular/part_N.json`, where N is your annotator number:

    {"essays": {
      "120": {"regular": true, "pairs": [
        ["opening words of 1A", "closing words of 1A", "opening words of 1B", "closing words of 1B"],
        ...four pairs...]},
      "121": {"regular": false, "reason": "three pairs; no 束股"}
    }}

- Keys are essay indices as strings. Every essay of your range must appear.
- Opening and closing words are taken from the unpunctuated text (the
  characters exactly as printed, without punctuation; watch for variant
  characters). Give four to eight characters each; the opening words of leg A
  must occur only once in the essay, or `check` will ask for more characters.
  A leg runs from its opening words to the first occurrence of its closing
  words after them.
- If `check` reports that the two legs of a pair have different numbers of
  fragments (the automatic punctuation splits them differently), give that
  pair instead as {"a": "...", "b": "..."} with your own punctuation
  (，between fragments). Only punctuation may differ from the text: the
  characters must be exactly those of the essay, in order. Put the breaks
  where the parallel clauses end, so that corresponding fragments of A and B
  correspond in meaning.
- Save the file after every ten essays or so, so that work is not lost.
- When several annotators work at the same time, keep any helper scripts and
  working copies in a private folder with names of your own, and never write
  to another annotator's part file; shared temporary folders have let one
  annotator's script overwrite another's work.

## Before you finish

Run `check` on your file until it reports no problems, and read the printed
legs of a sample of regular essays to make sure each leg begins and ends
where you intended. Report: the number of essays read, the number regular,
the most common reasons for exclusion, and any essays you found hard to call.
