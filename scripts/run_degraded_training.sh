#!/usr/bin/env bash
# Fine-tune GujiGPT on degraded or enciphered couplets and score the models (in parallel).
#
# For every meter in $METERS and condition in $CONDITIONS:
#   1. make_degraded_training.py  builds the training corpus with degraded inner couplets
#      (recombined, wordswap, nonsense, shuffled), or make_cipher_couplets.py the enciphered
#      corpus and held-out sets (charcipher, wordcipher)
#   2. finetune_io_gpt.py         fine-tunes GujiGPT on it (as the original models: 1 epoch,
#                                 lr 2e-5, batch size 8), with --seed 0 (--no-convert for ciphers,
#                                 which are built in traditional characters)
#   3. score_degraded_models.py   scores the model on the held-out couplets
# Examples:
#   METERS=penta CONDITIONS="charcipher wordcipher" nohup caffeinate -is bash scripts/run_degraded_training.sh \
#       > logs/degraded_training/driver_cipher.log 2>&1 &
# The jobs run in parallel; when all are done, make_degraded_figure.py draws the profiles.
#
# Full run, surviving a dropped ssh connection and keeping the Mac awake:
#   mkdir -p logs/degraded_training
#   nohup caffeinate -is bash scripts/run_degraded_training.sh > logs/degraded_training/driver.log 2>&1 &
# Smoke test (64 training couplets, 16 test couplets per set, all outputs under $SMOKE_DIR):
#   SMOKE=1 SMOKE_DIR=/path/to/scratch bash scripts/run_degraded_training.sh

set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-venv/bin/python}"
METERS="${METERS:-penta hepta}"
CONDITIONS="${CONDITIONS:-recombined nonsense shuffled}"

if [[ "${SMOKE:-0}" == "1" ]]; then
    ROOT="${SMOKE_DIR:?set SMOKE_DIR for a smoke test}"
    TRAIN_LIMIT=(--limit 64)
    TEST_LIMIT=(--limit 16)
    DATA_DIR="$ROOT/data"; CIPHER_DIR="$ROOT/cipher"; MODEL_DIR="$ROOT/models"; RESULTS_DIR="$ROOT/results"; LOG_DIR="$ROOT/logs"
else
    TRAIN_LIMIT=(); TEST_LIMIT=()
    DATA_DIR="data/2-3_regulated_verse/degraded_train"; CIPHER_DIR="data/2-3_regulated_verse/cipher"; MODEL_DIR="models"; RESULTS_DIR="results/degraded_training"
    LOG_DIR="logs/degraded_training"
fi
mkdir -p "$DATA_DIR" "$CIPHER_DIR" "$MODEL_DIR" "$RESULTS_DIR" "$LOG_DIR"

job() {
    local meter=$1 condition=$2
    local model="$MODEL_DIR/gujigpt-io-finetuned-$condition-$meter"
    set -e
    echo "[$(date '+%F %T')] start $meter $condition"
    if [[ "$condition" == *cipher ]]; then
        "$PY" -u scripts/make_cipher_couplets.py --meter "$meter" --level "${condition%cipher}" \
            --output-dir "$CIPHER_DIR" ${TRAIN_LIMIT[@]+"${TRAIN_LIMIT[@]}"}
        "$PY" -u scripts/finetune_io_gpt.py --data-path "$CIPHER_DIR/${meter}_${condition}_train.json" \
            --output-dir "$model" --seed 0 --no-convert
    else
        if [[ "$condition" == wordswap && ! -f "data/2-3_regulated_verse/couplet_controls/${meter}_wordswap.json" ]]; then
            echo "missing data/2-3_regulated_verse/couplet_controls/${meter}_wordswap.json (make_couplet_controls.py wordswap)"; exit 1
        fi
        "$PY" -u scripts/make_degraded_training.py --meter "$meter" --condition "$condition" \
            --output-dir "$DATA_DIR" ${TRAIN_LIMIT[@]+"${TRAIN_LIMIT[@]}"}
        "$PY" -u scripts/finetune_io_gpt.py --data-path "$DATA_DIR/${meter}_${condition}.json" \
            --output-dir "$model" --seed 0
    fi
    "$PY" -u scripts/score_degraded_models.py --meter "$meter" --condition "$condition" --model "$model" \
        --cipher-dir "$CIPHER_DIR" --output "$RESULTS_DIR/decoding_${meter}_${condition}.json" \
        ${TEST_LIMIT[@]+"${TEST_LIMIT[@]}"}
    echo "[$(date '+%F %T')] done $meter $condition"
}

echo "[$(date '+%F %T')] driver pid $$; meters: $METERS; conditions: $CONDITIONS; logs in $LOG_DIR"
pids=(); names=()
for meter in $METERS; do
    for condition in $CONDITIONS; do
        ( job "$meter" "$condition" ) > "$LOG_DIR/${meter}_${condition}.log" 2>&1 &
        pids+=($!); names+=("${meter}_${condition}")
        echo "started ${meter}_${condition}: pid $!"
    done
done

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[$(date '+%F %T')] ${names[$i]} finished"
    else
        echo "[$(date '+%F %T')] ${names[$i]} FAILED (see $LOG_DIR/${names[$i]}.log)"
        failed=1
    fi
done

"$PY" -u scripts/make_degraded_figure.py --results-dir "$RESULTS_DIR"
echo "[$(date '+%F %T')] all done (failed: $failed)"
exit $failed
