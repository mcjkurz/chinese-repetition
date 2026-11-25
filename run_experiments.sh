#!/usr/bin/env bash
# Run selected Chinese-repetition experiment pipelines.
#
# Usage:
#   ./run_experiments.sh
#   ./run_experiments.sh --only bert
#   ./run_experiments.sh --only gpt-score,alignment
#   ./run_experiments.sh --only scoring
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

usage() {
  cat <<'EOF'
Usage: ./run_experiments.sh [--only PIPELINES]

PIPELINES is a comma-separated list:
  all             Prepare data, run GPT and all BERT conditions, memorization, alignment
  gpt             Fine-tune GPT, score GPT, and measure memorization
  gpt-train       Fine-tune GPT and shuffled-label controls
  gpt-score       Score existing GPT models (fine-tuned, shuffled, and base)
  gpt-base-no-prompt  Score untuned GPT without I/O condition tokens
  bert            Run base, fine-tuned, and shuffled-label BERT conditions
  bert-train      Fine-tune GujiBERT on penta and hepta couplets
  bert-score      Score existing fine-tuned GujiBERT models on held-out couplets
  bert-base-score Score untuned GujiBERT on held-out couplets
  bert-control    Fine-tune and score shuffled-label BERT controls
  bert-control-train  Fine-tune shuffled-label GujiBERT controls
  bert-control-score  Score existing shuffled-label BERT controls
  scoring         Score existing GPT and BERT models without fine-tuning
  prepare         Rebuild poem-level train/test couplet splits
  memorization    Measure GPT verbatim recall on train and held-out data
  alignment       Run GujiBERT transition-alignment analyses

Examples:
  ./run_experiments.sh --only bert
  ./run_experiments.sh --only bert-score
  ./run_experiments.sh --only gpt-score,alignment
  ./run_experiments.sh --only prepare,bert-train

When a selected train or score stage needs split files that are absent, they
are prepared automatically. Existing split files are otherwise left untouched.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "$#" -gt 2 || ( "$#" -gt 0 && "$1" != "--only" ) ]]; then
  usage >&2
  exit 2
fi
if [[ "$#" -eq 2 ]]; then
  REQUESTED="$2"
else
  REQUESTED="all"
fi
IFS=',' read -r -a REQUESTED_PIPELINES <<< "${REQUESTED}"

PIPELINES=()
for requested_pipeline in "${REQUESTED_PIPELINES[@]}"; do
  case "${requested_pipeline}" in
    all)
      PIPELINES+=(prepare gpt-train gpt-score bert-train bert-score bert-base-score bert-control-train bert-control-score bert-decompose memorization alignment)
      ;;
    gpt)
      PIPELINES+=(gpt-train gpt-score memorization)
      ;;
    bert)
      PIPELINES+=(bert-train bert-score bert-base-score bert-control-train bert-control-score bert-decompose)
      ;;
    scoring)
      PIPELINES+=(gpt-score bert-score bert-base-score bert-control-score)
      ;;
    bert-control)
      PIPELINES+=(bert-control-train bert-control-score bert-decompose)
      ;;
    gpt-train|gpt-score|gpt-base-no-prompt|bert-train|bert-score|bert-base-score|bert-control-train|bert-control-score|bert-decompose|prepare|memorization|alignment)
      PIPELINES+=("${requested_pipeline}")
      ;;
    *)
      echo "Unknown pipeline: ${requested_pipeline}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

has_pipeline() {
  local target="$1"
  local selected
  for selected in "${PIPELINES[@]}"; do
    if [[ "${selected}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ ! -x "${ROOT}/venv/bin/python" ]]; then
  echo "Missing ${ROOT}/venv. Create it and install requirements first:" >&2
  echo "  python3 -m venv venv && ./venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# Do not source the activate script: it may contain another machine's venv path.
export VIRTUAL_ENV="${ROOT}/venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
unset PYTHONHOME
hash -r

step() {
  echo
  echo "================================================================"
  echo "$*"
  echo "================================================================"
}

run_prepare() {
  step "Poem-level couplet splits (pentasyllabic and heptasyllabic)"
  python scripts/prepare_data_io.py
}

ensure_split_data() {
  local missing=0
  for meter_name in penta hepta; do
    for split_name in train test; do
      if [[ ! -f "data/2-3_regulated_verse/couplets_${meter_name}_${split_name}.json" ]]; then
        missing=1
      fi
    done
  done
  if [[ "${missing}" -eq 1 ]]; then
    echo "Split files are missing; preparing them before the selected pipeline."
    run_prepare
  fi
}

run_gpt_train() {
  ensure_split_data
  for meter_name in penta hepta; do
    step "Fine-tune GujiGPT on ${meter_name}"
    python scripts/finetune_io_gpt.py \
      --model-name hsc748NLP/GujiGPT_fan \
      --data-path "data/2-3_regulated_verse/couplets_${meter_name}_train.json" \
      --output-dir "models/gujigpt-io-finetuned-${meter_name}" \
      --epochs 1 \
      --batch-size 16 \
      --learning-rate 2e-5

    step "Fine-tune shuffled-label GujiGPT control on ${meter_name}"
    python scripts/finetune_io_gpt.py \
      --model-name hsc748NLP/GujiGPT_fan \
      --data-path "data/2-3_regulated_verse/couplets_${meter_name}_train.json" \
      --output-dir "models/gujigpt-io-finetuned-shuffled-${meter_name}" \
      --epochs 1 \
      --batch-size 16 \
      --learning-rate 2e-5 \
      --control
  done
}

run_gpt_score() {
  ensure_split_data
  for meter_code in 5 7; do
    if [[ "${meter_code}" == "5" ]]; then
      meter_name="penta"
    else
      meter_name="hepta"
    fi
    step "Score fine-tuned GujiGPT on ${meter_name}"
    python scripts/decoding_io.py \
      --test finetuned \
      --meters "${meter_code}" \
      --model-path "models/gujigpt-io-finetuned-${meter_name}" \
      --batch-size 8 \
      --run-name gujigpt_io_finetuned

    step "Score shuffled-label GujiGPT control on ${meter_name}"
    python scripts/decoding_io.py \
      --test finetuned \
      --meters "${meter_code}" \
      --model-path "models/gujigpt-io-finetuned-shuffled-${meter_name}" \
      --batch-size 8 \
      --run-name gujigpt_io_shuffled
  done
  step "Score untuned GujiGPT on both meters"
  python scripts/decoding_io.py \
    --test base-model \
    --batch-size 8 \
    --run-name gujigpt_base
}

run_gpt_base_no_prompt() {
  ensure_split_data
  step "Score untuned GujiGPT on both meters without I/O condition tokens"
  python scripts/decoding_io.py \
    --test base-model \
    --prompt-mode none \
    --batch-size 8 \
    --run-name gujigpt_base_no_prompt
}

run_bert_train() {
  ensure_split_data
  for meter_name in penta hepta; do
    step "Fine-tune GujiBERT on ${meter_name} (all second-line positions masked)"
    python scripts/finetune_io_bert.py \
      --model-name hsc748NLP/GujiBERT_fan \
      --data-path "data/2-3_regulated_verse/couplets_${meter_name}_train.json" \
      --output-dir "models/gujibert-io-masked-finetuned-${meter_name}" \
      --epochs 1 \
      --batch-size 16 \
      --learning-rate 2e-5
  done
}

run_bert_score() {
  ensure_split_data
  for meter_name in penta hepta; do
    step "Score masked GujiBERT on ${meter_name} held-out couplets"
    python scripts/score_io_bert.py \
      --model-path "models/gujibert-io-masked-finetuned-${meter_name}" \
      --data-path "data/2-3_regulated_verse/couplets_${meter_name}_test.json" \
      --batch-size 16 \
      --run-name gujibert_io_masked_finetuned
  done
}

run_bert_base_score() {
  ensure_split_data
  for meter_name in penta hepta; do
    step "Score untuned GujiBERT on ${meter_name} held-out couplets"
    python scripts/score_io_bert.py \
      --model-path hsc748NLP/GujiBERT_fan \
      --data-path "data/2-3_regulated_verse/couplets_${meter_name}_test.json" \
      --batch-size 16 \
      --run-name gujibert_base
  done
}

run_bert_control_train() {
  ensure_split_data
  for meter_name in penta hepta; do
    step "Fine-tune shuffled-label GujiBERT control on ${meter_name}"
    python scripts/finetune_io_bert.py \
      --model-name hsc748NLP/GujiBERT_fan \
      --data-path "data/2-3_regulated_verse/couplets_${meter_name}_train.json" \
      --output-dir "models/gujibert-io-masked-finetuned-shuffled-${meter_name}" \
      --epochs 1 \
      --batch-size 16 \
      --learning-rate 2e-5 \
      --control \
      --shuffle-seed 42
  done
}

run_bert_control_score() {
  ensure_split_data
  for meter_name in penta hepta; do
    step "Score shuffled-label GujiBERT control on ${meter_name} held-out couplets"
    python scripts/score_io_bert.py \
      --model-path "models/gujibert-io-masked-finetuned-shuffled-${meter_name}" \
      --data-path "data/2-3_regulated_verse/couplets_${meter_name}_test.json" \
      --batch-size 16 \
      --run-name gujibert_io_shuffled
  done
}

run_bert_decompose() {
  step "Decompose GujiBERT inner-outer entropy gaps"
  python scripts/bert_entropy_gap_decomposition.py
}

run_memorization() {
  ensure_split_data
  for meter_code in 5 7; do
    if [[ "${meter_code}" == "5" ]]; then
      meter_name="penta"
    else
      meter_name="hepta"
    fi
    step "Measure verbatim recall for fine-tuned GujiGPT on ${meter_name}"
    python scripts/measure_memorization.py \
      --model-path "models/gujigpt-io-finetuned-${meter_name}" \
      --meters "${meter_code}" \
      --max-samples 10000 \
      --batch-size 8 \
      --run-name "finetuned_${meter_name}"
  done
  step "Measure verbatim recall for untuned GujiGPT"
  python scripts/measure_memorization.py \
    --model-path hsc748NLP/GujiGPT_fan \
    --max-samples 10000 \
    --batch-size 8 \
    --run-name base
}

run_alignment() {
  step "Alignment couplets (both meters)"
  python scripts/prepare_data_alignment.py
  step "GujiBERT transition alignment (both meters)"
  python scripts/transition_analysis.py \
    --model hsc748NLP/GujiBERT_fan \
    --model-label GujiBERT \
    --max-samples 100000 \
    --batch-size 32 \
    --output-dir results/transition_gujibert_paper \
    --baseline-correction subtract \
    --override
}

has_pipeline prepare && run_prepare
has_pipeline gpt-train && run_gpt_train
has_pipeline gpt-score && run_gpt_score
has_pipeline gpt-base-no-prompt && run_gpt_base_no_prompt
has_pipeline bert-train && run_bert_train
has_pipeline bert-score && run_bert_score
has_pipeline bert-base-score && run_bert_base_score
has_pipeline bert-control-train && run_bert_control_train
has_pipeline bert-control-score && run_bert_control_score
has_pipeline bert-decompose && run_bert_decompose
has_pipeline memorization && run_memorization
has_pipeline alignment && run_alignment

echo
echo "Selected pipelines complete: ${REQUESTED}"
echo "GPT decoding outputs: results/gpt_decoding/"
echo "GPT memorization outputs: results/gpt_memorization/"
echo "Masked-line BERT outputs: results/bert_masked_line/"
