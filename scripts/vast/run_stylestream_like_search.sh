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

: "${OUT_DIR:=runs/stylestream_like_search}"
: "${NUM_SOURCES:=20}"
: "${PAIRS_PER_SOURCE:=5}"
: "${PRESET:=default}"
: "${PAIRING:=sample}"
: "${KIND:=vevotimbre}"

: "${MAX_PAIRS:=50}"
: "${FINAL_MAX_PAIRS:=0}"
: "${KEEP_FIRST_N_AUDIO:=5}"

: "${WHISPER_MODEL:=base}"
: "${FINAL_WHISPER_MODEL:=large-v3}"
: "${USE_ACCENT:=0}"
: "${USE_EMOTION:=0}"

: "${FLOW_STEPS_GRID:=8,12,16,24,32}"
: "${DIFFUSION_CFG_GRID:=0.8,1.0,1.2}"
: "${DIFFUSION_RESCALE_GRID:=0.0,0.5,0.75}"

python -m evaluation.stylestream_like.build_manifest \
  --out_dir "$OUT_DIR" \
  --num_sources "$NUM_SOURCES" \
  --preset "$PRESET" \
  --pairing "$PAIRING" \
  --pairs_per_source "$PAIRS_PER_SOURCE"

ARGS=()
if [[ "$USE_ACCENT" == "1" ]]; then ARGS+=(--use_accent); fi
if [[ "$USE_EMOTION" == "1" ]]; then ARGS+=(--use_emotion); fi

python -m evaluation.stylestream_like.search \
  --manifest "$OUT_DIR/manifest.json" \
  --kind "$KIND" \
  --out_dir "$OUT_DIR/search_${KIND}" \
  --max_pairs "$MAX_PAIRS" \
  --final_max_pairs "$FINAL_MAX_PAIRS" \
  --keep_first_n_audio "$KEEP_FIRST_N_AUDIO" \
  --whisper_model "$WHISPER_MODEL" \
  --final_whisper_model "$FINAL_WHISPER_MODEL" \
  --flow_steps_grid "$FLOW_STEPS_GRID" \
  --diffusion_cfg_grid "$DIFFUSION_CFG_GRID" \
  --diffusion_rescale_grid "$DIFFUSION_RESCALE_GRID" \
  "${ARGS[@]}"
