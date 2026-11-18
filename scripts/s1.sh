#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
gpu="$1"
shift
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu"
export MUJOCO_GL=egl PYTHONPATH=src:../Movens-Interact OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=../Movens-Interact/.local/train-env/bin/python
test "$(cat runs/m4_s0_pretrained/exit)" = 0
test "$(cat runs/m4_s0_cold/exit)" = 0
test "$(cat runs/m5_probe_v3_fixed/exit)" = 0
"$py" -B - <<'PY'
import json
from pathlib import Path
for arm in ("pretrained", "cold"):
    p = Path(f"runs/m4_s0_{arm}/evaluation.json")
    r = json.loads(p.read_text())
    clean = r["test_splits"]["clean"]
    if (r["env_steps"] != 1721 or clean["episodes"] != [7, 16] or r["updates"] <= 0
        or clean["complete"] != 2 or clean["engineering_errors"] or r["train"]["complete"] != 2):
        raise ValueError(f"Incomplete S0 acceptance: {p}")
p = Path("runs/m5_probe_v3_fixed/verification.json")
r = json.loads(p.read_text())
if (r["actual_20ms_steps"] != 90 or r["physical_profile"]["observation_delay_ticks"] != 1
    or not .0199 < r["max_encoder_age_s"] < .0201 or not r["missing_frames"] or not r["visible_frames"]):
    raise ValueError(f"Incomplete deployment input check: {p}")
PY
# Evaluations use the same frozen checkpoint and can run in separate screens by mode.
exec "$py" -u -B -m movens_sim2real.learn \
  --pretrained runs/m4_s0_pretrained/final.pt --clean-replay runs/m4_s0_pretrained \
  --deployment curriculum --out runs/m5_s1 --train 1 8 2 4 --eval \
  --steps 3428 --max-updates 480 --warmup 500 --update-ratio .15 --batch 64 --seed 42 --save-every 32 "$@"
