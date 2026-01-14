#!/usr/bin/env bash
set -euo pipefail

# Runs ChatterboxVC over the deterministic French playlist, then scores the outputs.

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge3}"
if [[ ! -f "${MINIFORGE_ROOT}/etc/profile.d/conda.sh" ]]; then
  if command -v conda >/dev/null 2>&1; then
    MINIFORGE_ROOT="$(conda info --base)"
  fi
fi

CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[chatterbox_run_fleurs_fr_playlist] Missing conda.sh at ${CONDA_SH}. Set MINIFORGE_ROOT or install conda." >&2
  exit 1
fi

ENV_NAME="${ENV_NAME:-chatterbox}"

cd "$(dirname "$0")/../.."

PLAYLIST_DIR="${PLAYLIST_DIR:-data/vc_quest_playlists/fleurs_fr_fr_dev_v1}"
MANIFEST="${MANIFEST:-${PLAYLIST_DIR}/manifest.json}"

RUN_NAME="${RUN_NAME:-chatterbox_w800_h400_s8}"
RUN_DIR="runs/vc_quest/playlists/fleurs_fr_fr/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

CHATTERBOX_DIR="${HOME}/deps/chatterbox"
SEED="${SEED:-0}"
CFM_TIMESTEPS="${CFM_TIMESTEPS:-8}"
WATERMARK="${WATERMARK:-1}"

# Reuse the setup script's cache location to avoid re-downloading ~1GB checkpoints.
HF_HOME_DIR="${HF_HOME_DIR:-${HOME}/.hf_home}"
mkdir -p "${HF_HOME_DIR}"
export HF_HOME="${HF_HOME_DIR}"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-800}"
STREAM_HOP_MS="${STREAM_HOP_MS:-400}"
STREAM_FADE_MS="${STREAM_FADE_MS:-10}"

EMIT_ALIGN="${EMIT_ALIGN:-center}"

VAD_MODE="${VAD_MODE:-webrtc}"
VAD_DB="${VAD_DB:--55}"
VAD_FRAME_MS="${VAD_FRAME_MS:-10}"
VAD_HANGOVER_MS="${VAD_HANGOVER_MS:-200}"
VAD_WEBRTC_AGGRESSIVENESS="${VAD_WEBRTC_AGGRESSIVENESS:-2}"
VAD_WEBRTC_FRAME_MS="${VAD_WEBRTC_FRAME_MS:-30}"
VAD_WEBRTC_MIN_VOICED_RATIO="${VAD_WEBRTC_MIN_VOICED_RATIO:-0.1}"

GAIN_MODE="${GAIN_MODE:-match_src_rms}"
GAIN_TARGET_DELTA_DB="${GAIN_TARGET_DELTA_DB:-10.0}"
GAIN_MAX_BOOST_DB="${GAIN_MAX_BOOST_DB:-18.0}"
GAIN_SMOOTHING="${GAIN_SMOOTHING:-0.0}"

MASK_MODE="${MASK_MODE:-rms}"
MASK_DB="${MASK_DB:--50.0}"
MASK_FRAME_MS="${MASK_FRAME_MS:-10.0}"
MASK_SMOOTH_MS="${MASK_SMOOTH_MS:-10.0}"

PEAK_LIMIT="${PEAK_LIMIT:-0.99}"

MAX_PAIRS="${MAX_PAIRS:-0}"

# 1) Ensure playlist exists (build in current python env; uses HF downloads).
if [[ ! -f "${MANIFEST}" ]]; then
  scripts/vc_quest/fleurs_fr_build_playlist.sh
fi

# 2) Convert in the Chatterbox conda env.
source "${CONDA_SH}"
conda activate "${ENV_NAME}"

if [[ -z "${MODEL_DEVICE:-}" ]]; then
  MODEL_DEVICE="$(python - <<'PY'
import torch
if torch.cuda.is_available():
    print("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
)"
fi

WATERMARK_ARGS=()
if [[ "${WATERMARK}" == "0" ]]; then
  WATERMARK_ARGS+=(--no-watermark)
else
  WATERMARK_ARGS+=(--watermark)
fi

python -m evaluation.vc_quest.chatterbox_playlist_convert \
  --manifest "${MANIFEST}" \
  --out_dir "${RUN_DIR}" \
  --chatterbox_dir "${CHATTERBOX_DIR}" \
  --device "${MODEL_DEVICE}" \
  --seed "${SEED}" \
  --cfm_timesteps "${CFM_TIMESTEPS}" \
  "${WATERMARK_ARGS[@]}" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}" \
  --emit_align "${EMIT_ALIGN}" \
  --max_pairs "${MAX_PAIRS}" \
  --vad_mode "${VAD_MODE}" \
  --vad_db "${VAD_DB}" \
  --vad_frame_ms "${VAD_FRAME_MS}" \
  --vad_hangover_ms "${VAD_HANGOVER_MS}" \
  --vad_webrtc_aggressiveness "${VAD_WEBRTC_AGGRESSIVENESS}" \
  --vad_webrtc_frame_ms "${VAD_WEBRTC_FRAME_MS}" \
  --vad_webrtc_min_voiced_ratio "${VAD_WEBRTC_MIN_VOICED_RATIO}" \
  --gain_mode "${GAIN_MODE}" \
  --gain_target_delta_db "${GAIN_TARGET_DELTA_DB}" \
  --gain_max_boost_db "${GAIN_MAX_BOOST_DB}" \
  --gain_smoothing "${GAIN_SMOOTHING}" \
  --mask_mode "${MASK_MODE}" \
  --mask_db "${MASK_DB}" \
  --mask_frame_ms "${MASK_FRAME_MS}" \
  --mask_smooth_ms "${MASK_SMOOTH_MS}" \
  --peak_limit "${PEAK_LIMIT}"

conda deactivate || true

# 3) Score in Amphion's eval venv.
source .venv/bin/activate

SCORE_ARGS=()
if [[ "${PITCH_METRICS:-0}" == "1" ]]; then
  SCORE_ARGS+=(--pitch_metrics)
fi

WER_MODE="${WER_MODE:-transcript}"

python -m evaluation.vc_quest.score_playlist \
  --manifest "${MANIFEST}" \
  --run_dir "${RUN_DIR}" \
  --out_json "${RUN_DIR}/summary.json" \
  --whisper_model base \
  --whisper_language fr \
  --wer_mode "${WER_MODE}" \
  "${SCORE_ARGS[@]+"${SCORE_ARGS[@]}"}"

echo "[chatterbox_run_fleurs_fr_playlist] Wrote ${RUN_DIR}/summary.json"
