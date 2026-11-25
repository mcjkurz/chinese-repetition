# Leg alignment in regular eight-legged essays: method

This note fixes the procedure for measuring how closely the partner legs of
an eight-legged essay (八股文) move in parallel, so that the analysis can be
repeated on new data. It is the essay counterpart of the transition alignment
of regulated-verse couplets (`scripts/transition_analysis.py`).

## Data: regular essays only

Only regular essays enter the analysis, by analogy with regulated verse, whose
poems all have four couplets. An essay is regular when its body consists of
exactly four leg pairs (起股, 中股, 後股, 束股), that is eight legs, between the
opening sections (破題, 承題, 起講, 入題) and the conclusion (收結). This
selection has to be stated in the paper, since many essays of the later
periods do not keep all four pairs and are therefore left out.

- Source: 欽定四書文 (`data/4_eight_legged_essays/qinding_sishuwen.json`, 766 essays, five periods),
  text with automatic punctuation.
- Every essay is read, and for each one the annotator records either the four
  pairs or the reason why the essay is not regular (for example three pairs,
  five pairs, no 束股, legs not answering each other).
- A leg may be of any length, down to a single clause; no length threshold is
  applied.
- The two legs of a pair are given by their opening and closing words in the
  unpunctuated text. When the automatic punctuation splits the two legs into
  different numbers of fragments, the pair is given with hand punctuation
  instead, whose characters must match the essay text exactly.
- Sentences that join two pairs (出題, 過接) belong to no leg.
- Chen Zilong's essay (no. 346) takes its legs from the edited text
  `data/4_eight_legged_essays/baguwen_chen_zilong.txt`.

Target: 100 regular essays (400 pairs), or at least 80 essays (320 pairs).

Outcome (annotation of 2026-11-01, eight annotators, every essay read; brief
in `docs/baguwen_regular_annotation_brief.md`, reasons for exclusion in
`data/4_eight_legged_essays/qinding_regular/part_*.json`): 152 of the 766 essays are regular,
608 pairs, of which 2 have one-fragment legs.

| Period | Essays | Regular |
|---|---|---|
| 化治 | 57 | 4 |
| 正嘉 | 108 | 20 |
| 隆萬 | 100 | 15 |
| 啟禎 | 206 | 52 |
| 本朝 | 295 | 61 |

Boundary rules the annotators applied, to be kept in any extension:

- A short parallel inside the 起講, or a one-sentence antithesis in the
  收結 (often after 吁, 是則, 是知, 吾故曰), is not a leg pair; a full pair
  right after the 後股 is the 束股.
- A short opening pair right after the 起講 counts as the 起股 when an 出題
  sentence turning to the topic follows it.
- A short couplet that restates the topic between pairs (出題, 過接) is not a
  pair; a full fifth pair that carries content makes the essay irregular.
- The most common reasons for exclusion are three pairs (usually no 束股),
  one or two pairs or a body in two large halves (兩扇), continuous prose,
  five or more pairs, and bodies built on triads or lists of items.

The earlier sample of 100 essays (`data/4_eight_legged_essays/qinding_leg_annotations.json`) is
not a check on this annotation: it marked only long pairs, and several of
its essays with four or more pairs had one pair cut into parallel sentences.

## Fragments

Each leg is cut into fragments at ，。；：？！、. Every fragment is encoded on
its own by GujiBERT (`hsc748NLP/GujiBERT_fan`) and represented, at every
layer, by the mean of its characters' embeddings. Encoding fragments
separately keeps the length of a leg from shaping its representation, since a
long leg is never encoded as one sequence. Every character must be a single
GujiBERT token.

## Measure 1: within a pair (fragment level)

The couplet analogue: a pair plays the part of a couplet, a leg the part of a
line, and a fragment the part of a character.

1. Take the transitions between consecutive fragments inside leg A
   (fragment 1→2, 2→3, …) and inside leg B, each normalised to unit length.
2. Take the cosine between corresponding transitions of A and B (A's 1→2
   against B's 1→2, and so on) and average them: one value per pair.
3. Average the four pairs with equal weight: one value per essay. Also report
   each pair position (起股, 中股, 後股, 束股) separately, as couplet positions
   are reported for the poems.

A pair needs at least two fragments per leg and the same number of fragments
in both legs. A pair of one-fragment legs has no value; such essays are
reported on their remaining pairs, and the number of pairs without a value is
stated.

## Measure 2: between pairs (leg level)

1. Represent each leg by the mean of its fragment vectors.
2. Odd stream: legs 1→3→5→7; even stream: legs 2→4→6→8; three transitions
   each.
3. Take the cosine between corresponding transitions (1→3 against 2→4, 3→5
   against 4→6, 5→7 against 6→8) and average them: one value per essay.

This measure covers one-fragment legs and does not need partner legs to have
the same number of fragments. It rests on three transitions per essay and is
therefore noisier than measure 1. Leg means carry much of an essay's theme,
so a large part of the raw value can come from the shared topic; the baseline
below is what separates parallel movement from shared topic. The poems have no
direct counterpart (it would compare line 1 → line 3 with line 2 → line 4
across couplets), which should be said in the paper.

## Baselines

- Measure 1, random pairs: leg A of a pair against leg B of random pairs with
  the same number of fragments, from other essays and from pseudo-leg pairs of
  古文觀止 (2n consecutive fragments of one prose piece, split into two
  pseudo-legs of n fragments, fragment lengths within the range of the real
  ones, no fragment repeated immediately).
- Measure 2, random essays: the odd stream of an essay against the even stream
  of random other essays; prose baseline from eight consecutive pseudo-legs of
  古文觀止 with the fragment counts of the real legs.

## Controls

Both controls are built on every regular essay, pair by pair, so that each
control essay can be compared with the real essay it was built on.

- Composed (`scripts/make_qinding_composed.py`): legs written by hand on a
  different saying of the Four Books, with the same number of fragments, every
  fragment within ±2 characters of the real one, no run of four characters
  copied from the real essay, one GujiBERT token per character, and no
  fragment repeated immediately within a leg.
- Nonsense (`scripts/make_qinding_nonsense.py`, since 2026-11-04): every pair
  of corresponding fragments (fragment k of leg A and of leg B) replaced by a
  pair of corresponding fragments taken whole from a leg pair of another
  essay, a different essay for every pair of fragments, so that every
  vertical pair is a real parallel pair while no leg, and no stretch of the
  essay, makes sense (the essay counterpart of the segment-swapped couplets).
  Donor fragments have the lengths of the fragments they replace wherever an
  unused essay offers them (2,462 of 2,512 fragment pairs), otherwise the
  closest lengths; no leg repeats a fragment immediately. The earlier
  character-substitution controls (every distinct character replaced by a
  random frequent character of the Mengzi) remain available with
  `--method characters`; their files and results are kept in
  `data/4_eight_legged_essays/qinding_regular100/backup_charsub_nonsense/` and
  `results/baguwen_regular100/backup_charsub_nonsense_2026-11-04/`.

## Statistics

The essay is the unit. For each condition and layer: mean of the essay values
with a bootstrap 95% confidence interval over essays; Mann-Whitney test and
Cohen's d against the 古文觀止 baseline; Wilcoxon signed-rank test of each real
essay against the control built on it. The figure layer is 8, as elsewhere.
Measure 1 is the main measure, and measure 2 is reported alongside it as a
check that includes the movement from one pair to the next.

## Design: four conditions of equal size

- 欽定四書文: 100 of the 152 regular essays (400 pairs), sampled as evenly
  across periods as the data allow (`scripts/sample_regular_essays.py`, seed
  0): all regular essays of 化治 (4), 正嘉 (20) and 隆萬 (15), and 30 of 啟禎
  (always including Chen Zilong, no. 346) and 31 of 本朝.
- Composed: one composed essay per real essay, four pairs each, written by
  hand after `docs/baguwen_composed_brief.md`; each essay argues another
  saying of the Four Books.
- Nonsense: one per real essay, from corresponding fragment pairs of other
  essays.
- 古文觀止: one pseudo-essay per real essay (`scripts/make_pseudo_essays.py`):
  consecutive fragments of one piece cut into eight pseudo-legs with the
  fragment counts of the real essay's legs, so that prose has exactly the
  pairs, legs and fragments of the real essays. Measure 1 reads its
  pseudo-pairs (1A-1B, 2A-2B, ...), measure 2 its eight legs.

The random-pair baseline of measure 1 draws on a larger sample of 古文觀止
pseudo-leg pairs (`--baseline-pool`, up to 300 per fragment count), so that
pairs of rare fragment counts still have enough random partners.

## Reproduction

1. Annotate every essay: `data/4_eight_legged_essays/qinding_regular/part_*.json`, after
   `docs/baguwen_regular_annotation_brief.md`, checked with
   `scripts/check_regular_legs.py check <file>`.
2. Merge and build: `scripts/check_regular_legs.py merge`, then
   `scripts/prepare_qinding_legs.py --annotations data/4_eight_legged_essays/qinding_regular_annotations.json --min-pairs 4 --min-fragments 1 --min-chars 1 --output data/4_eight_legged_essays/qinding_regular_legs.json`.
3. Sample: `scripts/sample_regular_essays.py` (writes
   `data/4_eight_legged_essays/qinding_regular100/legs.json`). Every leg must have at least two
   clauses (`--min-fragments 2`), since a single-clause leg has no clause
   transition; the two sampled essays that failed this (419, 492) were
   replaced by 309 and 605, so that the sample has 400 measurable pairs.
4. Controls and prose:
   - composed: writers' files `data/4_eight_legged_essays/qinding_regular100/composed_part_*.txt`
     (part 9 holds the essays 309 and 605 that replaced 419 and 492),
     joined into `data/4_eight_legged_essays/qinding_regular100/composed.txt` and built with
     `scripts/make_qinding_composed.py --source data/4_eight_legged_essays/qinding_regular100/composed.txt --corpus data/4_eight_legged_essays/qinding_regular100/legs.json --output data/4_eight_legged_essays/qinding_regular100/composed.json`;
   - nonsense: `scripts/make_qinding_nonsense.py --corpus data/4_eight_legged_essays/qinding_regular100/legs.json --output data/4_eight_legged_essays/qinding_regular100/nonsense.json`;
   - prose: `scripts/make_pseudo_essays.py`.
5. Measure 1: `scripts/baguwen_pseudo_legs.py --corpus "欽定四書文=data/4_eight_legged_essays/qinding_regular100/legs.json"
   --controls "Composed=data/4_eight_legged_essays/qinding_regular100/composed.json" "Nonsense=data/4_eight_legged_essays/qinding_regular100/nonsense.json"
   --prose "古文觀止=data/4_eight_legged_essays/qinding_regular100/guwen_pseudo.json" --min-fragments 2 --dots texts
   --output-dir results/baguwen_regular100/pair_alignment`. Panel (c) of the
   figure (Chen Zilong's fragments in one PCA, legs 1, 3, 5, 7 beside legs 2,
   4, 6, 8 on the first two components) is drawn in the same run; add
   `--replot` to redraw the figure from the saved results, and
   `--essay-dims 3 --essay-view ELEV AZIM` for the three-dimensional version
   in two cubes.
6. Measure 2: `scripts/baguwen_leg_streams.py` (defaults point to the same
   files; output in `results/baguwen_regular100/leg_streams`).
