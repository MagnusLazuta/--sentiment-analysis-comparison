#!/bin/bash
#SBATCH --job-name=pln-tf
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# Usage (from ~/pln-tf/experimento):
#   sbatch slurm/run_all.sbatch                         # full run
#   LIMIT=20 sbatch slurm/run_all.sbatch                # pilot, 20 items per dataset
#   RUN_ID=<id> sbatch slurm/run_all.sbatch             # resume an earlier run
#   MODELS_TO_RUN="kev-4b laya" sbatch ...              # subset of models
#   ABLATION_OF=<run id> sbatch slurm/run_all.sbatch    # context-length ablation of that run
#   ABLATION_OF=<run id> ITEMS_MAKER=make_extra.py sbatch slurm/run_all.sbatch  # follow-ups A/B
set -euo pipefail
export PROJ="${SLURM_SUBMIT_DIR:-$PWD}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-local}}"
[ -n "${SCRATCH:-}" ] || { echo "SCRATCH not set"; exit 1; }
echo "RUN_ID=$RUN_ID"

# Optional gate: GATE_JOB=<pilot job id> (submit with --dependency=afterany:<id>).
# Runs only if that pilot finished every model with zero item failures and its
# evaluation and report succeeded.
if [ -n "${GATE_JOB:-}" ]; then
  st=$(ls -d "$HOME"/pln-tf/results/*-"$GATE_JOB"/logs/status.txt 2>/dev/null | head -1)
  [ -n "$st" ] || { echo "GATE: no status.txt for pilot job $GATE_JOB; not running"; exit 1; }
  cat "$st"
  bad=$(grep -v -E " ok\(item-failures=0\) " "$st" || true)
  n_ok=$(grep -c -E " ok\(item-failures=0\) " "$st" || true)
  if [ -n "$bad" ] || [ "$n_ok" -lt 7 ]; then
    echo "GATE: pilot $GATE_JOB not clean (ok models: $n_ok/7); not running the full study"; exit 1
  fi
  echo "GATE: pilot $GATE_JOB clean; starting the full run"
fi

if command -v nix >/dev/null 2>&1; then
  exec nix --extra-experimental-features "nix-command flakes" develop "path:$PROJ" \
    --command bash "$PROJ/slurm/pipeline.sh"
fi
# fallback without nix: standalone uv in $SCRATCH; bsdtar/git/curl from the system
if ! command -v uv >/dev/null 2>&1; then
  export UV_INSTALL_DIR="$SCRATCH/pln-tf/bin"
  [ -x "$UV_INSTALL_DIR/uv" ] || curl -LsSf https://astral.sh/uv/0.12.19/install.sh | sh
  export PATH="$UV_INSTALL_DIR:$PATH"
fi
exec bash "$PROJ/slurm/pipeline.sh"
