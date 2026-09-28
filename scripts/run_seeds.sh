#!/usr/bin/env bash
# Train one config over several seeds, then print mean ± std of the test accuracy.
#
#   bash scripts/run_seeds.sh <shd|ssc|gsc|smnist> <config.yaml> [extra train args...]
#   SEEDS="0 1 2 3 4" bash scripts/run_seeds.sh gsc configs/gsc/fits_o1_w512.yaml --data-dir /data/GSC/processed
#
# Runs go to <results_dir>_seed<k>; set CUDA_VISIBLE_DEVICES to pick the GPU.
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,7p' "$0"; exit 1
fi
TASK="$1"; CONFIG="$2"; shift 2
SEEDS="${SEEDS:-0 1 2 3 4}"

cd "$(dirname "$0")/.."
BASE=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['results_dir'])" "$CONFIG")

DIRS=()
for SEED in $SEEDS; do
    OUT="${BASE}_seed${SEED}"
    echo "=== ${TASK} | ${CONFIG} | seed ${SEED} -> ${OUT}"
    python -m "experiments.${TASK}.train" --config "$CONFIG" --seed "$SEED" --results-dir "$OUT" "$@"
    DIRS+=("$OUT")
done
python scripts/summarize.py "${DIRS[@]}"
