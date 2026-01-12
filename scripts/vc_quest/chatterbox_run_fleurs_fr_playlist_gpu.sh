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
WATERMARK="${WATERMARK:-1}"

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
  --device cuda:0 \
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
  "${SCORE_ARGS[@]}"

echo "[chatterbox_run_fleurs_fr_playlist] Wrote ${RUN_DIR}/summary.json"
