#!/usr/bin/env bash
set -euo pipefail
project="$(cd "$(dirname "$0")/.." && pwd)"
index="${MOVENS_REFIT_INDEX:-$project/runs/m6_export/index.json}"
context="${MOVENS_REFIT_CONTEXT:-$project/runs/m6_context}"
export CUDA_VISIBLE_DEVICES="$1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
shift
cd "$project/../Movens-Interact"
exec .local/train-env/bin/python -u -B -m training.cache \
  --manifest "$index" --output "$context" \
  --splits train --device cuda:0 --batch-size 2 --chunk-frames 32 --gpu-memory-fraction .5 "$@"
