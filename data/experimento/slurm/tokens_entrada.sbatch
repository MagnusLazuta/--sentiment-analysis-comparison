#!/bin/bash
#SBATCH --job-name=pln-tf-tokens
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
# Input-token count per model (src/tokens_entrada.py). CPU only, tokenizers only (2026-11-23).
# Usage (from ~/pln-tf/experimento): RUN=20261117-181106-838446 sbatch slurm/tokens_entrada.sbatch
set -uo pipefail
PROJ="${SLURM_SUBMIT_DIR:-$PWD}"
RUN=${RUN:?}
WORK="$SCRATCH/pln-tf"
OUT="$HOME/pln-tf/results/tokens-${SLURM_JOB_ID:-local}"
export HF_HOME="$WORK/hf" UV_CACHE_DIR="$WORK/uv-cache" UV_PYTHON_INSTALL_DIR="$WORK/uv-python"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=""
if ! command -v uv >/dev/null 2>&1; then
  export UV_INSTALL_DIR="$WORK/bin"
  [ -x "$UV_INSTALL_DIR/uv" ] || curl -LsSf https://astral.sh/uv/0.12.19/install.sh | sh
  export PATH="$UV_INSTALL_DIR:$PATH"
fi
mkvenv() {  # name requirements (same pinned files as the main pipeline)
  local v="$WORK/venv-$1"
  [ -x "$v/bin/python" ] || { uv venv -q --python 3.12 "$v" && VIRTUAL_ENV="$v" uv pip install -q -r "$2" || { rm -rf "$v"; return 1; }; }
}
mkdir -p "$OUT"
hostname
ITEMS="$HOME/pln-tf/results/$RUN/data/items.jsonl"
echo "$RUN" > "$OUT/run.txt"
cd "$PROJ/src" || exit 1
# a model that fails is logged and skipped; the others still run
if mkvenv main "$PROJ/requirements/main.txt"; then
  "$WORK/venv-main/bin/python" tokens_entrada.py "$ITEMS" "$OUT/tokens_main.csv" \
    twitter-roberta deberta-nli qwen3.5-4b mistral-7b nanojev || echo "main models FAILED"
else
  echo "venv-main FAILED"
fi
if mkvenv laya "$PROJ/requirements/laya.txt"; then
  "$WORK/venv-laya/bin/python" tokens_entrada.py "$ITEMS" "$OUT/tokens_laya.csv" laya || echo "laya FAILED"
else
  echo "venv-laya FAILED"
fi
ls -la "$OUT"
