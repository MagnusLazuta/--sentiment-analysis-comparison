#!/bin/bash
# Whole study in one job: environments, data, models, inference, metrics, report.
# Called by run_all.sbatch. Heavy files live in $SCRATCH; results are copied to $RESULTS.
#
# Failure policy: only the shared setup (main env, data) may stop the job. Any model
# (its environment, download or inference) that fails is recorded in logs/status.txt
# and skipped. Re-submitting with the same RUN_ID reruns only what is missing:
# finished predictions are copied back after every model and reused.
set -uo pipefail

PROJ=${PROJ:?}            # code (home, shared)
RUN_ID=${RUN_ID:?}
LIMIT=${LIMIT:-0}         # >0: pilot with LIMIT items per dataset
ABLATION_OF=${ABLATION_OF:-}  # RUN_ID of a finished run: context-length ablation of its items
ITEMS_MAKER=${ITEMS_MAKER:-make_ablation.py}  # with ABLATION_OF: builds the items (make_extra.py = follow-ups)
MODELS_TO_RUN=${MODELS_TO_RUN:-"twitter-roberta deberta-nli qwen3.5-4b mistral-7b kev-4b laya nanojev"}
WORK="$SCRATCH/pln-tf"
RUN="$WORK/runs/$RUN_ID"
RESULTS="$HOME/pln-tf/results/$RUN_ID"
SRC="$PROJ/src"
export HF_HOME="$WORK/hf" UV_CACHE_DIR="$WORK/uv-cache" UV_PYTHON_INSTALL_DIR="$WORK/uv-python"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN/preds" "$RUN/logs" "$RESULTS"
STATUS="$RUN/logs/status.txt"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "FATAL: $*"; echo "setup FAILED: $*" >> "$STATUS"; exit 1; }
status() { echo "$1 $2 $3" >> "$STATUS"; log "$1: $2 ($3)"; }  # model state seconds
sync_back() { rsync -a "$RUN/preds" "$RUN/logs" "$RESULTS/" 2>/dev/null || cp -r "$RUN/preds" "$RUN/logs" "$RESULTS/"; }
KEV_PID=""
cleanup() { [ -n "$KEV_PID" ] && pkill -P "$KEV_PID" 2>/dev/null; [ -n "$KEV_PID" ] && kill "$KEV_PID" 2>/dev/null; sync_back; }
trap cleanup EXIT

# reuse predictions from an earlier attempt of this run (may have been on another node)
[ -d "$RESULTS/preds" ] && cp -n "$RESULTS"/preds/*.jsonl "$RUN/preds/" 2>/dev/null

log "node $(hostname) job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$RUN/logs/gpu.txt" \
  || die "nvidia-smi failed: no GPU visible"

# ---- shared setup (fatal on failure) -----------------------------------------
mkvenv() {  # name requirements
  local v="$WORK/venv-$1"
  if [ ! -x "$v/bin/python" ]; then
    uv venv -q --python 3.12 "$v" && VIRTUAL_ENV="$v" uv pip install -q -r "$2" || { rm -rf "$v"; return 1; }
  fi
  VIRTUAL_ENV="$v" uv pip freeze > "$RUN/logs/freeze-$1.txt"
  "$v/bin/python" -c "import torch, sys; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
}
log "env main"
mkvenv main "$PROJ/requirements/main.txt" > "$RUN/logs/env-main.txt" 2>&1 \
  || die "main env (or CUDA in it) failed, see logs/env-main.txt"
PY="$WORK/venv-main/bin/python"

log "data"
cd "$SRC" || die "no $SRC"
if [ -n "$ABLATION_OF" ]; then
  mkdir -p "$RUN/data"
  "$PY" "$ITEMS_MAKER" "$HOME/pln-tf/results/$ABLATION_OF/data/items.jsonl" "$RUN/data/items.jsonl" \
    > "$RUN/logs/make_items.txt" 2>&1 || die "$ITEMS_MAKER"
  echo "$ABLATION_OF $ITEMS_MAKER" > "$RUN/logs/ablation_of.txt"
else
# Filatova ships .rar; without bsdtar use the official static 7-Zip 25.01 (sha256 pinned)
if ! command -v bsdtar >/dev/null 2>&1; then
  export SEVENZIP="$WORK/bin/7zzs"
  if [ ! -x "$SEVENZIP" ]; then
    mkdir -p "$WORK/bin" && curl -sfL -o "$WORK/bin/7z.tar.xz" https://www.7-zip.org/a/7z2501-linux-x64.tar.xz \
      && echo "4ca3b7c6f2f67866b92622818b58233dc70367be2f36b498eb0bdeaaa44b53f4  $WORK/bin/7z.tar.xz" | sha256sum -c - \
      && "$PY" -c "import tarfile,sys; tarfile.open(sys.argv[1]).extract('7zzs', sys.argv[2], filter='data')" \
           "$WORK/bin/7z.tar.xz" "$WORK/bin" && chmod +x "$SEVENZIP" || die "7-Zip download failed"
  fi
fi
"$PY" download_data.py "$WORK/data/raw" > "$RUN/logs/download_data.txt" 2>&1 || die "download_data"
"$PY" prepare_data.py "$WORK/data/raw" "$RUN/data" > "$RUN/logs/prepare_data.txt" 2>&1 || die "prepare_data"
cp "$WORK/data/raw/manifest.json" "$RUN/logs/data_manifest.json"
cp "$RUN/data/items_report.json" "$RUN/logs/"
fi
ITEMS="$RUN/data/items.jsonl"

# ---- per-model setup (non-fatal) ---------------------------------------------
clone() {  # repo commit dir
  { [ -d "$3/.git" ] || git clone -q "https://github.com/$1.git" "$3"; } && git -C "$3" checkout -q "$2"
}
cfg() { "$PY" -c "import sys; sys.path.insert(0, '$SRC'); from config import MODELS as M; print(M['$1']['$2'])"; }

log "models download"
"$PY" download_models.py "$WORK/nanojev-ckpt" "$RUN/logs/model_manifest.json" > "$RUN/logs/download_models.txt" 2>&1 \
  || log "WARNING: some model downloads failed (see logs/download_models.txt); those models will fail and be skipped"

setup_model() {  # returns non-zero if the model cannot run
  case $1 in
    kev-4b)
      clone "$(cfg kev-4b code_repo)" "$(cfg kev-4b code_commit)" "$WORK/kev" \
        && (cd "$WORK/kev" && uv sync -q --extra serve --frozen) ;;
    laya)
      mkvenv laya "$PROJ/requirements/laya.txt" ;;
    nanojev)
      clone "$(cfg nanojev code_repo)" "$(cfg nanojev code_commit)" "$WORK/nanojev" \
        && [ -f "$WORK/nanojev-ckpt/best.safetensors" ] ;;
    *) true ;;
  esac
}

start_kev() {
  (cd "$WORK/kev" && exec .venv/bin/python -m kev.serve --run "$(cfg kev-4b run_spec)" --port 8008) \
    > "$RUN/logs/kev-server.txt" 2>&1 &
  KEV_PID=$!
  for _ in $(seq 1 90); do
    kill -0 "$KEV_PID" 2>/dev/null || { log "kev server died"; return 1; }
    curl -sf http://127.0.0.1:8008/v1/models > "$RUN/logs/kev-models.json" && return 0
    sleep 10
  done
  log "kev server not ready after 15 min"; return 1
}
stop_kev() { [ -n "$KEV_PID" ] && { pkill -P "$KEV_PID" 2>/dev/null; kill "$KEV_PID" 2>/dev/null; wait "$KEV_PID" 2>/dev/null; }; KEV_PID=""; }

run_model() {
  local m=$1 out="$RUN/preds/$1.jsonl"
  case $m in
    twitter-roberta|deberta-nli)
      "$PY" run_hf.py --model "$m" --items "$ITEMS" --out "$out" --limit "$LIMIT" --batch 32 ;;
    qwen3.5-4b|mistral-7b)
      "$PY" run_hf.py --model "$m" --items "$ITEMS" --out "$out" --limit "$LIMIT" --batch 1 ;;  # no padding (see diag_padding.py)
    kev-4b)
      start_kev || return 1
      "$PY" run_systemone.py --model kev-4b --items "$ITEMS" --out "$out" --limit "$LIMIT"
      local rc=$?; stop_kev; return $rc ;;
    laya)
      "$WORK/venv-laya/bin/python" run_systemone.py --model laya --items "$ITEMS" --out "$out" --limit "$LIMIT" ;;
    nanojev)
      "$PY" run_systemone.py --model nanojev --items "$ITEMS" --out "$out" --limit "$LIMIT" \
        --code-dir "$WORK/nanojev" --ckpt-dir "$WORK/nanojev-ckpt" ;;
    *) log "unknown model $m"; return 1 ;;
  esac
}

# ---- inference ------------------------------------------------------------------
for m in $MODELS_TO_RUN; do
  start=$(date +%s)
  log "setup $m"
  if ! setup_model "$m" > "$RUN/logs/setup-$m.txt" 2>&1; then
    status "$m" SKIPPED-setup-failed $(( $(date +%s) - start )); sync_back; continue
  fi
  log "run $m"
  if run_model "$m" > "$RUN/logs/run-$m.txt" 2>&1; then
    nf=$(cat "$RUN/preds/$m.jsonl.failures.jsonl" 2>/dev/null | wc -l)
    status "$m" "ok(item-failures=$nf)" $(( $(date +%s) - start ))
  else
    status "$m" FAILED-see-run-log $(( $(date +%s) - start ))
  fi
  stop_kev
  sync_back
done

# ---- metrics and report (non-fatal) ------------------------------------------------
if [ -n "$ABLATION_OF" ]; then  # analysed locally against the source run
  cp -r "$RUN/data" "$RESULTS/" 2>/dev/null; log "done: $RESULTS"; cat "$STATUS"; exit 0
fi
log "evaluate"
"$PY" evaluate.py "$ITEMS" "$RUN/preds" "$RUN/eval" > "$RUN/logs/evaluate.txt" 2>&1 || status evaluate FAILED 0
log "report"
"$PY" report.py "$ITEMS" "$RUN/preds" "$RUN/eval" "$RUN/report" > "$RUN/logs/report.txt" 2>&1 || status report FAILED 0
cp -r "$RUN/eval" "$RUN/data" "$RUN/report" "$RESULTS/" 2>/dev/null
log "done: $RESULTS"
cat "$STATUS"
