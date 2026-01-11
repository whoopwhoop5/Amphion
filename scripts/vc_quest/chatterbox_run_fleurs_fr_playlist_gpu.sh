#!/usr/bin/env bash
set -euo pipefail

# Runs ChatterboxVC over the deterministic French playlist, then scores the outputs.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="chatterbox"

cd "$(dirname "$0")/../.."

PLAYLIST_DIR="${PLAYLIST_DIR:-data/vc_quest_playlists/fleurs_fr_fr_dev_v1}"
MANIFEST="${MANIFEST:-${PLAYLIST_DIR}/manifest.json}"

RUN_NAME="${RUN_NAME:-chatterbox_w800_h400_s8}"
RUN_DIR="runs/vc_quest/playlists/fleurs_fr_fr/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

CHATTERBOX_DIR="${HOME}/deps/chatterbox"
SEED="${SEED:-0}"
CFM_TIMESTEPS="${CFM_TIMESTEPS:-8}"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-800}"
STREAM_HOP_MS="${STREAM_HOP_MS:-400}"
STREAM_FADE_MS="${STREAM_FADE_MS:-10}"

# 1) Ensure playlist exists (build in current python env; uses HF downloads).
if [[ ! -f "${MANIFEST}" ]]; then
  scripts/vc_quest/fleurs_fr_build_playlist.sh
fi

# 2) Convert in the Chatterbox conda env.
source "${CONDA_SH}"
conda activate "${ENV_NAME}"

python -m evaluation.vc_quest.chatterbox_playlist_convert \
  --manifest "${MANIFEST}" \
  --out_dir "${RUN_DIR}" \
  --chatterbox_dir "${CHATTERBOX_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --cfm_timesteps "${CFM_TIMESTEPS}" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}"

conda deactivate || true

# 3) Score in Amphion's eval venv.
source .venv/bin/activate

python -m evaluation.vc_quest.score_playlist \
  --manifest "${MANIFEST}" \
  --run_dir "${RUN_DIR}" \
  --out_json "${RUN_DIR}/summary.json" \
  --whisper_model base \
  --whisper_language fr

echo "[chatterbox_run_fleurs_fr_playlist] Wrote ${RUN_DIR}/summary.json"

