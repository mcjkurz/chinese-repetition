# Brief for writing the composed controls of the regular essays

This brief is given to every writer, so that the composed controls can be
repeated or extended. The method they serve is
`docs/baguwen_leg_alignment_method.md`; the essays are the 100 regular essays
of `data/4_eight_legged_essays/qinding_regular100/legs.json` (four leg pairs each).

## Task

For every essay assigned to you, write four leg pairs in classical Chinese
that stand in for the essay's four real pairs: same shape, different content.
A composed essay is what a competent examination candidate might have
written on another topic, keeping the layout of the real essay.

- Choose for each essay one saying of the Four Books as its topic, different
  from the essay's own topic (its title), and argue that topic in all four
  pairs. Use the range of the Four Books assigned to you, so that writers do
  not pick the same sayings, and do not use one saying twice.
- Each composed pair must be a real pair of partner legs: leg B answers leg A
  clause by clause, with parallel syntax and matching or contrasting terms, as
  in the real pair it stands in for. Imitate the degree of parallelism of the
  real pair: if the real legs repeat many words, yours may too; if they are
  loosely parallel, yours should be loosely parallel.
- Write meaningful, grammatical classical prose in traditional characters,
  not filler or lists of words.

## Constraints (checked by the script)

For every leg of every pair, against the real leg it stands in for:

- the same number of fragments (fragments are separated by ，。；：？！、);
- every fragment within 2 characters of the length of the corresponding real
  fragment;
- no run of 4 or more characters copied from the real essay (text or legs);
- every character a single GujiBERT token (use common traditional forms;
  avoid rare variants);
- no fragment repeated immediately within a leg.

## Format

Write `data/4_eight_legged_essays/qinding_regular100/composed_part_N.txt`, where N is your writer
number:

    @ <essay id> <topic saying> (<book chapter.section>)
    A <leg A of the essay's first pair>
    B <leg B of the essay's first pair>
    A <leg A of the second pair>
    B ...

Four A/B pairs per essay, in the order of the essay's pairs. Lines starting
with # are comments. To see the real pairs, read the essay's entry in
`data/4_eight_legged_essays/qinding_regular100/legs.json` (its "pairs", each with "a", "b" and the
number of "fragments"). Check your file with:

    venv/bin/python scripts/make_qinding_composed.py --source data/4_eight_legged_essays/qinding_regular100/composed_part_N.txt \
        --corpus data/4_eight_legged_essays/qinding_regular100/legs.json --output <your private folder>/check.json

It reports every problem by line; it also reports the essays of the sample
that are "still missing", which is expected, since other writers cover them.
Repeat until it reports no problems for your essays.

## Working alongside other writers

Several writers work at the same time. Keep helper scripts and working copies
in a private folder with names of your own, and never write to another
writer's file. Save your file after every essay or two.

## Before you finish

Run the check until it reports no problems, reread a few of your pairs to
make sure they read as genuine partner legs, and report the essays done and
any that were hard to fit.
