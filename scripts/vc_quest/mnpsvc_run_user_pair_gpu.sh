#!/usr/bin/env bash
set -euo pipefail

# Runs MNP-SVC for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="mnpsvc"

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/mnpsvc/user_pair"
mkdir -p "${RUN_DIR}"

MNP_SVC_DIR="${HOME}/deps/mnpsvc"

MODEL_DEVICE="${MODEL_DEVICE:-cuda}"
SEED="${SEED:-0}"
REF_MAX_SEC="${REF_MAX_SEC:-10.0}"

PITCH_EXTRACTOR="${PITCH_EXTRACTOR:-rmvpe}"
F0_MIN="${F0_MIN:-50.0}"
F0_MAX="${F0_MAX:-1200.0}"
RESPONSE_THRESHOLD_DB="${RESPONSE_THRESHOLD_DB:--60.0}"

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

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

echo "[mnpsvc_run] repo=${MNP_SVC_DIR} device=${MODEL_DEVICE} ref_max_sec=${REF_MAX_SEC}"
echo "[mnpsvc_run] pitch_extractor=${PITCH_EXTRACTOR} f0_min=${F0_MIN} f0_max=${F0_MAX} response_th_db=${RESPONSE_THRESHOLD_DB}"
echo "[mnpsvc_run] Streaming: window=${STREAM_WINDOW_MS}ms hop=${STREAM_HOP_MS}ms emit_align=${STREAM_EMIT_ALIGN} vad=${VAD_MODE}"

python -m evaluation.vc_quest.mnpsvc_convert \
  --mnpsvc_dir "${MNP_SVC_DIR}" \
  --device "${MODEL_DEVICE}" \
  --seed "${SEED}" \
  --pitch_extractor "${PITCH_EXTRACTOR}" \
  --f0_min "${F0_MIN}" \
  --f0_max "${F0_MAX}" \
  --response_threshold_db "${RESPONSE_THRESHOLD_DB}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json"

python -m evaluation.vc_quest.mnpsvc_convert \
  --mnpsvc_dir "${MNP_SVC_DIR}" \
  --device "${MODEL_DEVICE}" \
  --seed "${SEED}" \
  --pitch_extractor "${PITCH_EXTRACTOR}" \
  --f0_min "${F0_MIN}" \
  --f0_max "${F0_MAX}" \
  --response_threshold_db "${RESPONSE_THRESHOLD_DB}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json"

python -m evaluation.vc_quest.mnpsvc_convert \
  --mnpsvc_dir "${MNP_SVC_DIR}" \
  --device "${MODEL_DEVICE}" \
  --seed "${SEED}" \
  --pitch_extractor "${PITCH_EXTRACTOR}" \
  --f0_min "${F0_MIN}" \
  --f0_max "${F0_MAX}" \
  --response_threshold_db "${RESPONSE_THRESHOLD_DB}" \
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

python -m evaluation.vc_quest.mnpsvc_convert \
  --mnpsvc_dir "${MNP_SVC_DIR}" \
  --device "${MODEL_DEVICE}" \
  --seed "${SEED}" \
  --pitch_extractor "${PITCH_EXTRACTOR}" \
  --f0_min "${F0_MIN}" \
  --f0_max "${F0_MAX}" \
  --response_threshold_db "${RESPONSE_THRESHOLD_DB}" \
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
  --out_json "${RUN_DIR}/v5_to_fr_offline.report.json" \
  --whisper_model base \
  --whisper_language fr

python -m evaluation.vc_quest.score_outputs \
  --ref_wav "${REF_FR}" \
  --src_wav "${SRC_V5}" \
  --deg_wav "${RUN_DIR}/v5_to_fr_stream.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream.meta.json" \
  --out_json "${RUN_DIR}/v5_to_fr_stream.report.json" \
  --whisper_model base \
  --whisper_language fr

python -m evaluation.vc_quest.score_outputs \
  --ref_wav "${REF_V5}" \
  --src_wav "${SRC_FR}" \
  --deg_wav "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json" \
  --out_json "${RUN_DIR}/fr_to_v5_offline.report.json" \
  --whisper_model base \
  --whisper_language fr

python -m evaluation.vc_quest.score_outputs \
  --ref_wav "${REF_V5}" \
  --src_wav "${SRC_FR}" \
  --deg_wav "${RUN_DIR}/fr_to_v5_stream.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream.meta.json" \
  --out_json "${RUN_DIR}/fr_to_v5_stream.report.json" \
  --whisper_model base \
  --whisper_language fr

echo "[mnpsvc_run] Wrote reports under ${RUN_DIR}"

