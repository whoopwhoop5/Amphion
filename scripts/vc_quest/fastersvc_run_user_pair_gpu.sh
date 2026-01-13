#!/usr/bin/env bash
set -euo pipefail

# Runs FasterSVC for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="fastersvc"

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/fastersvc/user_pair"
mkdir -p "${RUN_DIR}"

FASTER_SVC_DIR="${HOME}/deps/fastersvc"

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

MODEL_DEVICE="${MODEL_DEVICE:-cuda}"
SEED="${SEED:-0}"
REF_MAX_SEC="${REF_MAX_SEC:-10.0}"

KNN_K="${KNN_K:-4}"
KNN_ALPHA="${KNN_ALPHA:-0.0}"
PITCH_SHIFT="${PITCH_SHIFT:-0.0}"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-800}"
STREAM_HOP_MS="${STREAM_HOP_MS:-400}"
STREAM_FADE_MS="${STREAM_FADE_MS:-10}"
STREAM_EMIT_ALIGN="${STREAM_EMIT_ALIGN:-end}"

VAD_MODE="${VAD_MODE:-rms}"
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

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

echo "[fastersvc_run] repo=${FASTER_SVC_DIR} device=${MODEL_DEVICE} ref_max_sec=${REF_MAX_SEC}"
echo "[fastersvc_run] k=${KNN_K} alpha=${KNN_ALPHA} pitch_shift=${PITCH_SHIFT}"
echo "[fastersvc_run] Streaming: window=${STREAM_WINDOW_MS}ms hop=${STREAM_HOP_MS}ms emit_align=${STREAM_EMIT_ALIGN}"

python -m evaluation.vc_quest.fastersvc_convert \
  --fastersvc_dir "${FASTER_SVC_DIR}" \
  --device "${MODEL_DEVICE}" \
  --seed "${SEED}" \
  --k "${KNN_K}" \
  --alpha "${KNN_ALPHA}" \
  --pitch_shift "${PITCH_SHIFT}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json"

python -m evaluation.vc_quest.fastersvc_convert \
  --fastersvc_dir "${FASTER_SVC_DIR}" \
  --device "${MODEL_DEVICE}" \
  --seed "${SEED}" \
  --k "${KNN_K}" \
  --alpha "${KNN_ALPHA}" \
  --pitch_shift "${PITCH_SHIFT}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json"

python -m evaluation.vc_quest.fastersvc_convert \
  --fastersvc_dir "${FASTER_SVC_DIR}" \
  --device "${MODEL_DEVICE}" \
  --seed "${SEED}" \
  --k "${KNN_K}" \
  --alpha "${KNN_ALPHA}" \
  --pitch_shift "${PITCH_SHIFT}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --gain_mode "${GAIN_MODE}" \
  --gain_target_delta_db "${GAIN_TARGET_DELTA_DB}" \
  --gain_max_boost_db "${GAIN_MAX_BOOST_DB}" \
  --gain_smoothing "${GAIN_SMOOTHING}" \
  --mask_mode "${MASK_MODE}" \
  --mask_db "${MASK_DB}" \
  --mask_frame_ms "${MASK_FRAME_MS}" \
  --mask_smooth_ms "${MASK_SMOOTH_MS}" \
  --peak_limit "${PEAK_LIMIT}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_stream.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}" \
  --emit_align "${STREAM_EMIT_ALIGN}" \
  --vad_mode "${VAD_MODE}" \
  --vad_db "${VAD_DB}" \
  --vad_frame_ms "${VAD_FRAME_MS}" \
  --vad_hangover_ms "${VAD_HANGOVER_MS}" \
  --vad_webrtc_aggressiveness "${VAD_WEBRTC_AGGRESSIVENESS}" \
  --vad_webrtc_frame_ms "${VAD_WEBRTC_FRAME_MS}" \
  --vad_webrtc_min_voiced_ratio "${VAD_WEBRTC_MIN_VOICED_RATIO}"

python -m evaluation.vc_quest.fastersvc_convert \
  --fastersvc_dir "${FASTER_SVC_DIR}" \
  --device "${MODEL_DEVICE}" \
  --seed "${SEED}" \
  --k "${KNN_K}" \
  --alpha "${KNN_ALPHA}" \
  --pitch_shift "${PITCH_SHIFT}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --gain_mode "${GAIN_MODE}" \
  --gain_target_delta_db "${GAIN_TARGET_DELTA_DB}" \
  --gain_max_boost_db "${GAIN_MAX_BOOST_DB}" \
  --gain_smoothing "${GAIN_SMOOTHING}" \
  --mask_mode "${MASK_MODE}" \
  --mask_db "${MASK_DB}" \
  --mask_frame_ms "${MASK_FRAME_MS}" \
  --mask_smooth_ms "${MASK_SMOOTH_MS}" \
  --peak_limit "${PEAK_LIMIT}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_stream.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}" \
  --emit_align "${STREAM_EMIT_ALIGN}" \
  --vad_mode "${VAD_MODE}" \
  --vad_db "${VAD_DB}" \
  --vad_frame_ms "${VAD_FRAME_MS}" \
  --vad_hangover_ms "${VAD_HANGOVER_MS}" \
  --vad_webrtc_aggressiveness "${VAD_WEBRTC_AGGRESSIVENESS}" \
  --vad_webrtc_frame_ms "${VAD_WEBRTC_FRAME_MS}" \
  --vad_webrtc_min_voiced_ratio "${VAD_WEBRTC_MIN_VOICED_RATIO}"

conda deactivate || true
source .venv/bin/activate

python -m evaluation.vc_quest.score_outputs \
  --ref_wav "${REF_FR}" \
  --src_wav "${SRC_V5}" \
  --deg_wav "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json" \
  --out_json "${RUN_DIR}/v5_to_fr_offline.report.json"

python -m evaluation.vc_quest.score_outputs \
  --ref_wav "${REF_V5}" \
  --src_wav "${SRC_FR}" \
  --deg_wav "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json" \
  --out_json "${RUN_DIR}/fr_to_v5_offline.report.json"

python -m evaluation.vc_quest.score_outputs \
  --ref_wav "${REF_FR}" \
  --src_wav "${SRC_V5}" \
  --deg_wav "${RUN_DIR}/v5_to_fr_stream.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream.meta.json" \
  --out_json "${RUN_DIR}/v5_to_fr_stream.report.json"

python -m evaluation.vc_quest.score_outputs \
  --ref_wav "${REF_V5}" \
  --src_wav "${SRC_FR}" \
  --deg_wav "${RUN_DIR}/fr_to_v5_stream.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream.meta.json" \
  --out_json "${RUN_DIR}/fr_to_v5_stream.report.json"

echo "[fastersvc_run] Wrote artifacts to ${RUN_DIR}"
