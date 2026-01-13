#!/usr/bin/env bash
set -euo pipefail

# Runs a trained RVC model over a mixed playlist (FLEURS fr_fr sources + LibriTTS target ref), then scores.
#
# Prereqs:
#   bash scripts/vc_quest/rvc_setup_gpu.sh
#   bash scripts/vc_quest/rvc_train_libritts_speaker_gpu.sh

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="rvc"

cd "$(dirname "$0")/../.."

# Source playlist (French).
FLEURS_PLAYLIST_DIR="${FLEURS_PLAYLIST_DIR:-data/vc_quest_playlists/fleurs_fr_fr_dev_v1}"
FLEURS_MANIFEST="${FLEURS_MANIFEST:-${FLEURS_PLAYLIST_DIR}/manifest.json}"

# Target speaker dataset (LibriTTS export manifest).
LIBRITTS_SPLIT="${LIBRITTS_SPLIT:-train.clean.100}"
SEED="${SEED:-1234}"
LIBRITTS_DATASET_DIR="${LIBRITTS_DATASET_DIR:-runs/vc_quest/rvc/datasets/libritts_${LIBRITTS_SPLIT}_seed${SEED}}"
LIBRITTS_TARGET_MANIFEST="${LIBRITTS_TARGET_MANIFEST:-${LIBRITTS_DATASET_DIR}/manifest.json}"

# Mixed playlist output.
MIXED_PLAYLIST_DIR="${MIXED_PLAYLIST_DIR:-}"
MIXED_MANIFEST="${MIXED_MANIFEST:-}"
NUM_SOURCES="${NUM_SOURCES:-30}"

# RVC model paths.
RVC_DIR="${RVC_DIR:-${HOME}/deps/rvc_webui}"
MODEL_NAME="${MODEL_NAME:-}" # default computed from speaker_id after reading manifest
INDEX_PATH="${INDEX_PATH:-}"
INDEX_RATE="${INDEX_RATE:-0.0}" # recommended 0 for streaming (upstream reloads index per window)

F0METHOD="${F0METHOD:-rmvpe}"
F0UP_KEY="${F0UP_KEY:-0}"
FILTER_RADIUS="${FILTER_RADIUS:-3}"
RESAMPLE_SR="${RESAMPLE_SR:-0}"
RMS_MIX_RATE="${RMS_MIX_RATE:-1.0}"
PROTECT="${PROTECT:-0.33}"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-800}"
STREAM_HOP_MS="${STREAM_HOP_MS:-400}"
STREAM_FADE_MS="${STREAM_FADE_MS:-10}"
EMIT_ALIGN="${EMIT_ALIGN:-end}"
NORMALIZE_ALIGN="${NORMALIZE_ALIGN:-end}"

VAD_MODE="${VAD_MODE:-rms}"
VAD_DB="${VAD_DB:--55}"
VAD_FRAME_MS="${VAD_FRAME_MS:-10}"
VAD_HANGOVER_MS="${VAD_HANGOVER_MS:-200}"
VAD_WEBRTC_AGGRESSIVENESS="${VAD_WEBRTC_AGGRESSIVENESS:-2}"
VAD_WEBRTC_FRAME_MS="${VAD_WEBRTC_FRAME_MS:-30}"
VAD_WEBRTC_MIN_VOICED_RATIO="${VAD_WEBRTC_MIN_VOICED_RATIO:-0.1}"

MASK_MODE="${MASK_MODE:-off}"
MASK_DB="${MASK_DB:--50.0}"
MASK_FRAME_MS="${MASK_FRAME_MS:-10.0}"
MASK_SMOOTH_MS="${MASK_SMOOTH_MS:-10.0}"
PEAK_LIMIT="${PEAK_LIMIT:-0.99}"

MAX_PAIRS="${MAX_PAIRS:-0}"

# 1) Ensure FLEURS playlist exists (built in current python env; uses HF downloads).
if [[ ! -f "${FLEURS_MANIFEST}" ]]; then
  scripts/vc_quest/fleurs_fr_build_playlist.sh
fi

# 2) Read target speaker_id and ensure mixed playlist exists.
if [[ ! -f "${LIBRITTS_TARGET_MANIFEST}" ]]; then
  echo "[rvc_run_mixed] Missing LibriTTS target manifest: ${LIBRITTS_TARGET_MANIFEST}" >&2
  echo "[rvc_run_mixed] Run: bash scripts/vc_quest/rvc_train_libritts_speaker_gpu.sh" >&2
  exit 1
fi

SPEAKER_ID="$(python3 - <<PY
import json
from pathlib import Path
m = json.loads(Path("${LIBRITTS_TARGET_MANIFEST}").read_text(encoding="utf-8"))
print(str((m.get("meta") or {}).get("speaker_id") or "").strip())
PY
)"
if [[ -z "${SPEAKER_ID}" ]]; then
  echo "[rvc_run_mixed] Failed to read speaker_id from ${LIBRITTS_TARGET_MANIFEST}" >&2
  exit 1
fi

if [[ -z "${MODEL_NAME}" ]]; then
  MODEL_NAME="rvc_libritts_${LIBRITTS_SPLIT}_s${SPEAKER_ID}_v1_40k_f0_rmvpe.pth"
fi

if [[ -z "${MIXED_PLAYLIST_DIR}" ]]; then
  MIXED_PLAYLIST_DIR="data/vc_quest_playlists/fleurs_fr_fr_to_libritts_s${SPEAKER_ID}_v1"
fi
if [[ -z "${MIXED_MANIFEST}" ]]; then
  MIXED_MANIFEST="${MIXED_PLAYLIST_DIR}/manifest.json"
fi

if [[ ! -f "${MIXED_MANIFEST}" ]]; then
  python3 -m evaluation.vc_quest.playlists.build_fleurs_to_libritts_target_playlist \
    --fleurs_manifest "${FLEURS_MANIFEST}" \
    --libritts_target_manifest "${LIBRITTS_TARGET_MANIFEST}" \
    --seed "${SEED}" \
    --num_sources "${NUM_SOURCES}" \
    --out_dir "${MIXED_PLAYLIST_DIR}"
fi

# 3) Convert in the RVC conda env.
RUN_NAME="${RUN_NAME:-rvc_libritts_s${SPEAKER_ID}_on_fleurs_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}_${EMIT_ALIGN}}"
RUN_DIR="runs/vc_quest/playlists/fleurs_fr_fr/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

python -m evaluation.vc_quest.rvc_playlist_convert \
  --manifest "${MIXED_MANIFEST}" \
  --out_dir "${RUN_DIR}" \
  --rvc_dir "${RVC_DIR}" \
  --model_name "${MODEL_NAME}" \
  --device cuda:0 \
  --half \
  --f0up_key "${F0UP_KEY}" \
  --f0method "${F0METHOD}" \
  --index_path "${INDEX_PATH}" \
  --index_rate "${INDEX_RATE}" \
  --filter_radius "${FILTER_RADIUS}" \
  --resample_sr "${RESAMPLE_SR}" \
  --rms_mix_rate "${RMS_MIX_RATE}" \
  --protect "${PROTECT}" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}" \
  --normalize_align "${NORMALIZE_ALIGN}" \
  --emit_align "${EMIT_ALIGN}" \
  --max_pairs "${MAX_PAIRS}" \
  --vad_mode "${VAD_MODE}" \
  --vad_db "${VAD_DB}" \
  --vad_frame_ms "${VAD_FRAME_MS}" \
  --vad_hangover_ms "${VAD_HANGOVER_MS}" \
  --vad_webrtc_aggressiveness "${VAD_WEBRTC_AGGRESSIVENESS}" \
  --vad_webrtc_frame_ms "${VAD_WEBRTC_FRAME_MS}" \
  --vad_webrtc_min_voiced_ratio "${VAD_WEBRTC_MIN_VOICED_RATIO}" \
  --mask_mode "${MASK_MODE}" \
  --mask_db "${MASK_DB}" \
  --mask_frame_ms "${MASK_FRAME_MS}" \
  --mask_smooth_ms "${MASK_SMOOTH_MS}" \
  --peak_limit "${PEAK_LIMIT}"

conda deactivate || true

# 4) Score in Amphion's eval venv.
source .venv/bin/activate

SCORE_ARGS=()
if [[ "${PITCH_METRICS:-0}" == "1" ]]; then
  SCORE_ARGS+=(--pitch_metrics)
fi

WER_MODE="${WER_MODE:-transcript}"

python -m evaluation.vc_quest.score_playlist \
  --manifest "${MIXED_MANIFEST}" \
  --run_dir "${RUN_DIR}" \
  --out_json "${RUN_DIR}/summary.json" \
  --whisper_model base \
  --whisper_language fr \
  --wer_mode "${WER_MODE}" \
  "${SCORE_ARGS[@]}"

echo "[rvc_run_mixed] Wrote ${RUN_DIR}/summary.json"
