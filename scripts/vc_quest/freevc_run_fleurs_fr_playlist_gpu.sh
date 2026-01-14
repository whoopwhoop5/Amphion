#!/usr/bin/env bash
set -euo pipefail

# Runs FreeVC over the deterministic French playlist, then scores the outputs.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="freevc"

cd "$(dirname "$0")/../.."

PLAYLIST_DIR="${PLAYLIST_DIR:-data/vc_quest_playlists/fleurs_fr_fr_dev_v1}"
MANIFEST="${MANIFEST:-${PLAYLIST_DIR}/manifest.json}"

RUN_NAME="${RUN_NAME:-freevc_w800_h400}"
RUN_DIR="runs/vc_quest/playlists/fleurs_fr_fr/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

FREEVC_DIR="${HOME}/deps/FreeVC"
VARIANT="${VARIANT:-freevc-24}"
SEED="${SEED:-0}"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-800}"
STREAM_HOP_MS="${STREAM_HOP_MS:-400}"
STREAM_FADE_MS="${STREAM_FADE_MS:-10}"

EMIT_ALIGN="${EMIT_ALIGN:-center}"
STREAM="${STREAM:-1}"

VAD_MODE="${VAD_MODE:-webrtc}"
VAD_DB="${VAD_DB:--55}"
VAD_FRAME_MS="${VAD_FRAME_MS:-10}"
VAD_HANGOVER_MS="${VAD_HANGOVER_MS:-200}"
VAD_WEBRTC_AGGRESSIVENESS="${VAD_WEBRTC_AGGRESSIVENESS:-2}"
VAD_WEBRTC_FRAME_MS="${VAD_WEBRTC_FRAME_MS:-30}"
VAD_WEBRTC_MIN_VOICED_RATIO="${VAD_WEBRTC_MIN_VOICED_RATIO:-0.1}"

GAIN_MODE="${GAIN_MODE:-off}"
GAIN_TARGET_DELTA_DB="${GAIN_TARGET_DELTA_DB:-10.0}"
GAIN_MAX_BOOST_DB="${GAIN_MAX_BOOST_DB:-18.0}"
GAIN_SMOOTHING="${GAIN_SMOOTHING:-0.0}"

MASK_MODE="${MASK_MODE:-off}"
MASK_DB="${MASK_DB:--50.0}"
MASK_FRAME_MS="${MASK_FRAME_MS:-10.0}"
MASK_SMOOTH_MS="${MASK_SMOOTH_MS:-10.0}"

PEAK_LIMIT="${PEAK_LIMIT:-0.99}"

MAX_PAIRS="${MAX_PAIRS:-0}"

# 1) Ensure playlist exists (build in current python env; uses HF downloads).
if [[ ! -f "${MANIFEST}" ]]; then
  scripts/vc_quest/fleurs_fr_build_playlist.sh
fi

# 2) Convert in the FreeVC conda env.
source "${CONDA_SH}"
conda activate "${ENV_NAME}"

STREAM_ARGS=()
if [[ "${STREAM}" == "1" ]]; then
  STREAM_ARGS+=(--stream)
  STREAM_ARGS+=(--window_ms "${STREAM_WINDOW_MS}")
  STREAM_ARGS+=(--hop_ms "${STREAM_HOP_MS}")
  STREAM_ARGS+=(--fade_ms "${STREAM_FADE_MS}")
  STREAM_ARGS+=(--emit_align "${EMIT_ALIGN}")
  STREAM_ARGS+=(--vad_mode "${VAD_MODE}")
  STREAM_ARGS+=(--vad_db "${VAD_DB}")
  STREAM_ARGS+=(--vad_frame_ms "${VAD_FRAME_MS}")
  STREAM_ARGS+=(--vad_hangover_ms "${VAD_HANGOVER_MS}")
  STREAM_ARGS+=(--vad_webrtc_aggressiveness "${VAD_WEBRTC_AGGRESSIVENESS}")
  STREAM_ARGS+=(--vad_webrtc_frame_ms "${VAD_WEBRTC_FRAME_MS}")
  STREAM_ARGS+=(--vad_webrtc_min_voiced_ratio "${VAD_WEBRTC_MIN_VOICED_RATIO}")
  STREAM_ARGS+=(--gain_mode "${GAIN_MODE}")
  STREAM_ARGS+=(--gain_target_delta_db "${GAIN_TARGET_DELTA_DB}")
  STREAM_ARGS+=(--gain_max_boost_db "${GAIN_MAX_BOOST_DB}")
  STREAM_ARGS+=(--gain_smoothing "${GAIN_SMOOTHING}")
  STREAM_ARGS+=(--mask_mode "${MASK_MODE}")
  STREAM_ARGS+=(--mask_db "${MASK_DB}")
  STREAM_ARGS+=(--mask_frame_ms "${MASK_FRAME_MS}")
  STREAM_ARGS+=(--mask_smooth_ms "${MASK_SMOOTH_MS}")
  STREAM_ARGS+=(--peak_limit "${PEAK_LIMIT}")
fi

python -m evaluation.vc_quest.freevc_playlist_convert \
  --manifest "${MANIFEST}" \
  --out_dir "${RUN_DIR}" \
  --freevc_dir "${FREEVC_DIR}" \
  --variant "${VARIANT}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --max_pairs "${MAX_PAIRS}" \
  "${STREAM_ARGS[@]+"${STREAM_ARGS[@]}"}"

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

echo "[freevc_run_fleurs_fr_playlist] Wrote ${RUN_DIR}/summary.json"
