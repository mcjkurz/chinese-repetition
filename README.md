# Chinese Repetition Experiments

This repository contains the scripts for an article on repetition, predictability,
and meaning in classical Chinese prose and regulated poetry. The experiments use
GujiGPT for decoder-side surprisal and entropy, and GujiBERT for encoder-side
geometric structure.

## Setup

Run commands from the repository root:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Scripts write figures and JSON outputs to `results/`, grouped by experiment:
GPT decoding in `results/gpt_decoding/`, GPT memorization in
`results/gpt_memorization/`, masked-line GujiBERT results in
`results/bert_masked_line/`, and transition-alignment results in
`results/transition_gujibert_paper/` by default (or a named folder when
specified). Prose surprisal outputs are written to `results/surprisal/`;
ranked alignment examples are in `results/transition_alignment_examples/`.
Fine-tuned checkpoints are written to `models/`. Poem experiments read
pentasyllabic and heptasyllabic regulated verse with the poem text kept intact
so train and test can be split by poem.
`scripts/fetch_data.py` and `scripts/build_regulated_csv.py` rebuild that file
from the source CSVs.

## Data

`data/` is organized by the article's four experiments. Experiments 2 and 3
read the same poems, so they share one folder:

| Folder | Experiment | Contents |
|---|---|---|
| `data/1_mengzi_prose/` | 1. Decoding patterned prose | the *Mengzi* passage, the full *Mengzi* (character pool for the controls), and the composed, nonsense, and scrambled control passages |
| `data/2-3_regulated_verse/` | 2. Decoding and 3. encoding regulated verse | `regulated.csv` (all regulated poems), the held-out couplet test sets, the control couplets (composed, nonsense, shuffled) with their word lexicons, and the cipher maps and enciphered test sets |
| `data/4_eight_legged_essays/` | 4. Encoding the eight-legged essay | the *Qinding Sishu wen* essays with their leg annotations, the 100 regular essays with their composed, nonsense, and shuffled controls, the *Guwen guanzhi* pseudo-essays, and Chen Zilong's essay |
| `data/cipher_corpus/` | 3 and 4 (nonsense controls) | the slice of Daizhige on which GujiBERT's character embeddings are retrained (downloaded, see below) |

The written rules followed by the annotators and writers of the controls are
in `docs/`. `data/2-3_regulated_verse/regulated.csv` was built by
`scripts/fetch_data.py` and `scripts/build_regulated_csv.py` from the poem
collection they download; it is included so that the splits below can be
rebuilt from the same poems.

Large derived files are not included and are rebuilt by the scripts:

| File(s) | Rebuilt by |
|---|---|
| `data/2-3_regulated_verse/couplets_{penta,hepta}_train.json` | `scripts/prepare_data_io.py` |
| `data/2-3_regulated_verse/couplets_{penta,hepta}_alignment.json` | `scripts/prepare_data_alignment.py` |
| `data/2-3_regulated_verse/cipher/*_train.json` | `scripts/make_cipher_couplets.py` |
| `data/2-3_regulated_verse/degraded_train/` | `scripts/make_degraded_training.py` |
| `data/cipher_corpus/daizhige/raw/` and `results/cipher_bert/corpus/*.npy` | `scripts/cipher_bert/prepare.py` (downloads a slice of Daizhige) |

`scripts/cipher_finetune/` is self-contained: `finetune_cipher.py` reads the
training and test couplets from `couplets.json` beside it (see the README
there). The scripts in `scripts/cipher_bert/` retrain GujiBERT's character
embeddings on the enciphered corpus and measure alignment with the retrained
model.

## Results

`results/` holds the saved outputs from which the figures are drawn and the
reported statistics are taken. Fine-tuned models are not included;
`run_experiments.sh` writes them to `models/`.

## Figures

The article's data figures (1–4 and 6) are redrawn from saved results by one
script, which writes them to `figures/`:

```bash
./make_figures.sh
```

The header of `make_figures.sh` lists, for every figure, the plotting script,
its input files, and the script that produced each input. Plotting does not
train or rescore models, except that `scripts/model_comparison.py` scores any
Qwen3 curve missing from its cache and `scripts/make_trajectory_figure.py`
encodes its four example couplets with GujiBERT.

## 1. Single-Sequence Prose Surprisal

This workflow scores one supplied sequence with GujiGPT and plots its
next-token surprisal in nats. It does not create replacement or shuffled
conditions and does not calculate perplexity. It accepts a plain-text file or
text supplied directly on the command line.

The sample Mengzi sequence is `data/1_mengzi_prose/mengzi_surprisal.txt`. Use `--input` for
a UTF-8 `.txt` file, or `--text` for a literal sequence:

```bash
venv/bin/python scripts/sequence_surprisal.py \
  --input data/1_mengzi_prose/mengzi_surprisal.txt \
  --output-png results/surprisal/mengzi_surprisal.png \
  --output-json results/surprisal/mengzi_surprisal.json
```

The scorer writes both outputs under `results/surprisal/`; Figure 1 is drawn
from the JSON by `make_figures.sh`. A title is omitted by default; supply
`--title "Your title"` to add one.

```bash
venv/bin/python scripts/sequence_surprisal.py \
  --text "魚，我所欲也；熊掌，亦我所欲也。" \
  --output-png results/surprisal/example.png \
  --output-json results/surprisal/example.json \
  --title "Example sequence"
```

The first token has no preceding token in the supplied sequence and is therefore
left unscored. The earlier `scripts/context_replacement_plot.py` remains
available for reproducing earlier replacement experiments; the new sequence
scorer does not use it.

## 2. Couplet Decoding With GujiGPT

The second experiment fine-tunes GujiGPT on pentasyllabic and heptasyllabic
couplets labeled by position, then compares next-token entropy and observed
loss for inner and outer couplets under matched and mismatched prompts.

The held-out couplets are a poem-level split, built for both meters in one
run. All four couplets of a poem go to train or to test, and a test poem
shares no line with the training poems. `data/2-3_regulated_verse/split_report.json` records the
overlap checks.

```bash
venv/bin/python scripts/prepare_data_io.py
```

That writes `data/2-3_regulated_verse/couplets_penta_train.json`, `data/2-3_regulated_verse/couplets_penta_test.json`,
`data/2-3_regulated_verse/couplets_hepta_train.json`, and `data/2-3_regulated_verse/couplets_hepta_test.json`
(100,000 training couplets and 10,000 test couplets per meter). Pentasyllabic
and heptasyllabic verse are fine-tuned as two models. Each example is prefixed
with `i` or `o`.

```bash
venv/bin/python scripts/finetune_io_gpt.py \
  --model-name hsc748NLP/GujiGPT_fan \
  --data-path data/2-3_regulated_verse/couplets_penta_train.json \
  --output-dir models/gujigpt-io-finetuned-penta \
  --epochs 1 \
  --batch-size 16 \
  --learning-rate 2e-5

venv/bin/python scripts/finetune_io_gpt.py \
  --model-name hsc748NLP/GujiGPT_fan \
  --data-path data/2-3_regulated_verse/couplets_hepta_train.json \
  --output-dir models/gujigpt-io-finetuned-hepta \
  --epochs 1 \
  --batch-size 16 \
  --learning-rate 2e-5

venv/bin/python scripts/decoding_io.py \
  --test finetuned \
  --meters 5 \
  --model-path models/gujigpt-io-finetuned-penta \
  --batch-size 8 \
  --run-name gujigpt_io_finetuned_penta
```

Score heptasyllabic verse the same way with `--meters 7` and
`models/gujigpt-io-finetuned-hepta`. The untuned baseline is one model scored
on both meters. The shuffled-label control is the fine-tune command with
`--control` and its own output directory per meter. `./run_experiments.sh`
runs the full set, including the optional GujiBERT masked-line experiment.
Use `--only` to run a narrower pipeline without repeating unrelated work:

```bash
./run_experiments.sh --only bert          # Run base, fine-tuned, and shuffled-label BERT conditions
./run_experiments.sh --only bert-score    # Score existing fine-tuned BERT checkpoints
./run_experiments.sh --only bert-base-score # Score untuned GujiBERT as a baseline
./run_experiments.sh --only bert-control  # Run only shuffled-label BERT controls
./run_experiments.sh --only gpt-score     # Score existing GPT checkpoints only
./run_experiments.sh --only gpt-base-no-prompt # Score untuned GPT without I/O tokens
./run_experiments.sh --only scoring       # Score existing GPT and all BERT conditions
./run_experiments.sh --only alignment     # Run transition-alignment analyses
```

Other selectors include `all`, `gpt`, `gpt-train`, `bert-train`, `prepare`,
and `memorization`. The BERT pipeline covers untuned pretrained GujiBERT
(base), fine-tuned GujiBERT with meaningful I/O labels, and fine-tuned
GujiBERT with shuffled training labels. The base condition is scored directly
from `hsc748NLP/GujiBERT_fan`; it does not require training or a local checkpoint.
Selectors can be combined with commas, e.g.
`--only gpt-score,alignment`. Existing poem-level split files are reused;
missing split files are generated automatically for pipelines that need them.
For BERT fine-tuning, all three scoring conditions, and shuffled-label
decomposition, use `./run_experiments.sh --only bert`.
To compare untuned GPT predictions for inner and outer couplets without
condition labels, run `./run_experiments.sh --only gpt-base-no-prompt`.

### Optional GujiBERT masked-line comparison

The complementary masked-language-model experiment fine-tunes GujiBERT to
predict every character in the second line while the **entire second line is
masked**. At evaluation, it scores each position with the first line and I/O
prompt visible, but no second-line characters visible. This measures
first-line-based expectations rather than an unfolding continuation. The
exponentiated mean masked-character loss is pseudo-perplexity, not joint
sequence perplexity. Results are saved by default under
`results/bert_masked_line/`.

Train separate penta and hepta models, then score each on the held-out poems.
The `--only bert` pipeline trains and scores both fine-tuned conditions, scores
the untuned base model, and writes the shuffled-label entropy-gap decomposition.

```bash
venv/bin/python scripts/finetune_io_bert.py \
  --model-name hsc748NLP/GujiBERT_fan \
  --data-path data/2-3_regulated_verse/couplets_penta_train.json \
  --output-dir models/gujibert-io-masked-finetuned-penta \
  --epochs 1 \
  --batch-size 16 \
  --learning-rate 2e-5

venv/bin/python scripts/score_io_bert.py \
  --model-path models/gujibert-io-masked-finetuned-penta \
  --data-path data/2-3_regulated_verse/couplets_penta_test.json \
  --run-name gujibert_io_masked_finetuned
```

Repeat with `hepta` paths for the seven-character meter. Scoring evaluates
each held-out couplet under both I and O prompts, reports per-position entropy
and observed-character loss, and tests matched versus mismatched prompts.
Fine-tune a shuffled-label control with `--control` if comparing prompt effects
against the shuffled-label decomposition used in the GPT analysis. Use
`--only bert-base-score` to score only the untuned baseline, or
`--only bert-control` to train and score only the shuffled-label control.

GujiGPT decoding summaries and figures default to `results/gpt_decoding/`.
GPT recall summaries default to `results/gpt_memorization/`.

Verbatim memorization is measured separately. The script gives the model the
first line, greedy-decodes the second, and reports the exact-match rate and
the teacher-forced loss. Score the fine-tuned model and the untuned base
model on the same train and poem-held-out test couplets. A train-versus-test
gap is memorization; the test rate is what the held-out entropy comparison
can be accused of reflecting.

```bash
venv/bin/python scripts/measure_memorization.py \
  --model-path models/gujigpt-io-finetuned-penta \
  --meters 5 \
  --max-samples 10000 \
  --batch-size 8 \
  --run-name finetuned_penta

venv/bin/python scripts/measure_memorization.py \
  --model-path hsc748NLP/GujiGPT_fan \
  --max-samples 10000 \
  --batch-size 8 \
  --run-name base
```

`matched` uses the fine-tuning prefix. `none` uses the first line only. The
script stops if the two files still share a poem.

## 3. Couplet Geometry With GujiBERT

The third experiment treats each couplet as a trajectory through GujiBERT
embedding space. It measures whether the character-to-character transitions in
the first line align with those in the second line, comparing inner couplets
against outer couplets layer by layer.

Prepare the alignment data first. The default command samples 100,000
line-disjoint couplets per meter, equally across the four positions:

```bash
venv/bin/python scripts/prepare_data_alignment.py

venv/bin/python scripts/transition_analysis.py \
  --model hsc748NLP/GujiBERT_fan \
  --model-label GujiBERT \
  --max-samples 100000 \
  --batch-size 32 \
  --output-dir results/transition_gujibert_paper \
  --baseline-correction subtract \
  --override
```

Pass `--meters 5` or `--meters 7` to either script to run one meter. Outputs
are written separately for `penta` and `hepta`.

Optional 3D PCA version:

```bash
venv/bin/python scripts/transition_analysis.py \
  --model hsc748NLP/GujiBERT_fan \
  --model-label GujiBERT \
  --max-samples 100000 \
  --batch-size 32 \
  --pca-dims 3 \
  --output-dir results/transition_gujibert_pca3 \
  --override
```

To inspect examples, prepare Tang-dynasty inner couplets and rank them by
transition alignment:

```bash
venv/bin/python scripts/prepare_data_alignment.py \
  --meters 5 \
  --dynasty 唐 \
  --positions 2 3 \
  --all \
  --output-path-template data/2-3_regulated_verse/couplets_{meter}_alignment_tang.json

venv/bin/python scripts/ranked_transition_alignment.py \
  --model hsc748NLP/GujiBERT_fan \
  --model-label GujiBERT \
  --batch-size 32 \
  --data-path data/2-3_regulated_verse/couplets_penta_alignment_tang.json \
  --output-path results/transition_alignment_examples/ranked_transition_alignment_tang_positions_2_3.json
```

## 4. Prose Self-Similarity With GujiBERT

The final experiment applies the encoder to patterned prose. It plots a
character-level self-similarity matrix, making repeated or parallel semantic
blocks visible as off-diagonal structure.

```bash
venv/bin/python scripts/self_similarity_matrix.py \
  --model hsc748NLP/GujiBERT_fan \
  --input-file data/4_eight_legged_essays/baguwen.txt \
  --layer -1 \
  --output-png results/prose_similarity/self_similarity_matrix.png \
  --output-json results/prose_similarity/self_similarity_matrix.json
```
