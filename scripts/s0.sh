#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH=src:../Movens-Interact OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=../Movens-Interact/.local/train-env/bin/python
arm=$1
gpu=$2
shift 2
extra=()
case "$arm" in
  pretrained) ;;
  cold) extra+=(--cold) ;;
  *) printf 'Usage: %s pretrained|cold GPU [--resume]\n' "$0" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu"
exec "$py" -u -B -m movens_sim2real.learn --pretrained runs/m2_formal/latest.pt \
  --out "runs/m4_s0_$arm" --train 1 8 --eval 7 16 --eval-side --steps 1721 \
  --max-updates 256 --warmup 500 --batch 64 --update-ratio .15 --seed 42 --save-every 32 "${extra[@]}" "$@"
