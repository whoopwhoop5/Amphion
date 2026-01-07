#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Vast containers often mount a small `/workspace` volume and set `HF_HOME` there.
# Override to a roomier path to avoid "No space left on device" during downloads.
: "${HF_HOME:=/root/.cache/huggingface}"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/.cache}"

# shellcheck disable=SC1090
source "${VENV_DIR:-.venv}/bin/activate"

: "${OUT_DIR:=runs/stylestream_like}"
: "${NUM_SOURCES:=100}"
: "${PAIRS_PER_SOURCE:=10}"
: "${KIND:=vevotimbre}"
: "${FLOW_STEPS:=16}"
: "${WHISPER_MODEL:=large-v3}"
: "${USE_ACCENT:=1}"
: "${USE_EMOTION:=0}"
: "${MAX_PAIRS:=0}"

python -m evaluation.stylestream_like.build_manifest \
  --out_dir "$OUT_DIR" \
  --num_sources "$NUM_SOURCES" \
  --pairs_per_source "$PAIRS_PER_SOURCE"

ARGS=()
if [[ "$USE_ACCENT" == "1" ]]; then ARGS+=(--use_accent); fi
if [[ "$USE_EMOTION" == "1" ]]; then ARGS+=(--use_emotion); fi
if [[ "$MAX_PAIRS" != "0" ]]; then ARGS+=(--max_pairs "$MAX_PAIRS"); fi

python -m evaluation.stylestream_like.evaluate \
  --manifest "$OUT_DIR/manifest.json" \
  --kind "$KIND" \
  --out_dir "$OUT_DIR/eval_${KIND}" \
  --flow_matching_steps "$FLOW_STEPS" \
  --whisper_model "$WHISPER_MODEL" \
  "${ARGS[@]}"
