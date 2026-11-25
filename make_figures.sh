#!/usr/bin/env bash
# Redraw the article's five data figures (1-4, 6) from saved results and copy
# them to figures/. Figure 5 is a diagram, not data, and is not drawn here. No model is trained or rescored, except that
# model_comparison.py scores any Qwen3 curve missing from its cache and
# make_trajectory_figure.py encodes its four example couplets with GujiBERT.
#
# Figure  Plotting script (this file)          Inputs                                                    Produced by
# 1       scripts/model_comparison.py          data/1_mengzi_prose/mengzi_surprisal.txt                                 (source text)
#                                              data/1_mengzi_prose/surprisal_controls/sensical.json                     scripts/make_sensical_controls.py
#                                              data/1_mengzi_prose/surprisal_controls/items_random_composed_frame.json  scripts/make_item_nonsense_controls.py
#                                              data/1_mengzi_prose/surprisal_controls/scrambled.json                    scripts/make_scrambled_controls.py
#                                              results/surprisal/mengzi_surprisal.json                   scripts/sequence_surprisal.py
#                                              results/surprisal/controls_{sensical,nonsense_items,scrambled}.json
#                                                                                                        scripts/score_passages.py
#                                              results/surprisal/qwen_surprisal_curves.json              scripts/model_comparison.py (cache)
# 2       scripts/make_control_figures.py      results/couplet_controls/decoding_{penta,hepta}.json      scripts/score_couplet_controls.py
#                                              results/cipher_finetune/{meter}/{meter}_cipher_epoch{1,4}.json
#                                                                                                        scripts/cipher_finetune/finetune_cipher.py
# 3       scripts/make_trajectory_figure.py    four couplets named in the script                         (GujiBERT, run by the script)
# 4       scripts/make_control_figures.py      results/transition_gujibert_paper/transition_alignment_layer_position_matrix_baseline_subtracted_{meter}.json
#                                                                                                        scripts/transition_analysis.py
#                                              results/couplet_controls/alignment_{penta,hepta}.json     scripts/align_couplet_controls.py
#                                              results/cipher_bert/alignment/{penta,hepta}.json          scripts/cipher_bert/align_eval.py
# 6       scripts/baguwen_pseudo_legs.py       results/baguwen_regular100/pair_alignment/{summary,pairs}.json
#         (--replot)                                                                                     scripts/baguwen_pseudo_legs.py
#                                              results/cipher_bert/essays/pairs.json                     scripts/cipher_bert/align_essays.py
#
# Usage: ./make_figures.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
PY="${ROOT}/venv/bin/python"
FIG="figures"
mkdir -p "${FIG}"
export PYTHONPATH="${ROOT}/scripts"

echo "Figure 1"
"${PY}" scripts/model_comparison.py
cp results/surprisal/model_comparison_gujigpt_qwen3_8b.png "${FIG}/fig1-mengzi-models-surprisal.png"

echo "Figures 2 and 4"
"${PY}" scripts/make_control_figures.py
cp results/couplet_controls/entropy_by_position.png "${FIG}/fig2-entropy-by-position.png"
cp results/couplet_controls/alignment_combined.png "${FIG}/fig4-transition-alignment.png"

echo "Figure 3"
"${PY}" scripts/make_trajectory_figure.py --output "${FIG}/fig3-couplet-trajectories.png"

echo "Figure 6"
"${PY}" scripts/baguwen_pseudo_legs.py --replot \
  --output-dir results/baguwen_regular100/pair_alignment --no-essay-panel --dots texts \
  --drop Recombined --substitute "Nonsense=results/cipher_bert/essays/pairs.json" \
  --order 欽定四書文 Nonsense 古文觀止 Composed Shuffled --figure-name pseudo_legs_cipher.png
cp results/baguwen_regular100/pair_alignment/pseudo_legs_cipher.png "${FIG}/fig6-baguwen-leg-alignment.png"

echo "Done: ${FIG}/fig1-4, fig6"
