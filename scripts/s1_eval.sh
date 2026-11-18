#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
gpu="$1"
mode="$2"
case "$mode" in clean|physics|noisy|deployment) ;; *) exit 2 ;; esac
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" MUJOCO_GL=egl
export PYTHONPATH=src:../Movens-Interact OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test "$(cat runs/m5_s1/exit)" = 0
test -s runs/m5_s1/final.pt
exec ../Movens-Interact/.local/train-env/bin/python -u -B scripts/evaluate_policy.py \
  --out "runs/m5_eval_s1/$mode" --small runs/m5_s1/final.pt --episode 7 16 --seed 42 --modes "$mode"
