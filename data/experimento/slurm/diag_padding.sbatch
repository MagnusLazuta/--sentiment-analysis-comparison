#!/bin/bash
#SBATCH --job-name=pln-tf-diag
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
# Padding diagnostic for the two LLMs. Usage (from ~/pln-tf/experimento):
#   ITEMS=<path to items.jsonl> sbatch slurm/diag_padding.sbatch
set -uo pipefail
PROJ="${SLURM_SUBMIT_DIR:-$PWD}"
WORK="$SCRATCH/pln-tf"
export HF_HOME="$WORK/hf" UV_CACHE_DIR="$WORK/uv-cache" UV_PYTHON_INSTALL_DIR="$WORK/uv-python" PYTHONUNBUFFERED=1
if ! command -v uv >/dev/null 2>&1; then
  export UV_INSTALL_DIR="$WORK/bin"
  [ -x "$UV_INSTALL_DIR/uv" ] || curl -LsSf https://astral.sh/uv/0.12.19/install.sh | sh
  export PATH="$UV_INSTALL_DIR:$PATH"
fi
V="$WORK/venv-main"
if [ ! -x "$V/bin/python" ]; then
  uv venv -q --python 3.12 "$V" && VIRTUAL_ENV="$V" uv pip install -q -r "$PROJ/requirements/main.txt" || exit 1
fi
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
cd "$PROJ/src" || exit 1
for m in qwen3.5-4b mistral-7b; do
  "$V/bin/python" diag_padding.py "$ITEMS" "$m" 48 || echo "$m FAILED"
done
