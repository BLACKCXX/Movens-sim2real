#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:../Movens-Interact OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
py=../Movens-Interact/.local/train-env/bin/python
mkdir -p runs/m1
case "${1:-}" in
  apple) exec flock -n runs/m1/apple.lock "$py" -u -B -m movens_sim2real.data apple --out runs/m1 --workers 12 ;;
  sonic) exec flock -n runs/m1/sonic.lock "$py" -u -B -m movens_sim2real.data sonic --out runs/m1 ;;
  dense)
    # Frozen representative named groups. Test names are held out of training.
    exec flock -n runs/m1/dense.lock xargs -P 8 -n 1 "$py" -u -B -m movens_sim2real.data replay --out runs/m1/dense --episode       <<< "1 2 4 8 12 33 19 20 44 7 16 23"
    ;;
  *) printf 'Usage: %s apple|sonic|dense\n' "$0" >&2; exit 2 ;;
esac
