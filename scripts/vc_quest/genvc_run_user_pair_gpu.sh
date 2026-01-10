#!/usr/bin/env bash
set -euo pipefail

# Runs GenVC for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="genvc"

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/genvc/user_pair"
mkdir -p "${RUN_DIR}"

GENVC_DIR="${HOME}/deps/GenVC"

MODEL_VARIANT="${MODEL_VARIANT:-GenVC_small.pth}"
MODEL_PATH="${GENVC_DIR}/pre_trained/${MODEL_VARIANT}"

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

SEED="${SEED:-0}"
TOP_K="${TOP_K:-1}"

STREAM_CHUNK_MS="${STREAM_CHUNK_MS:-1000}"
STREAM_TOKEN_CHUNK="${STREAM_TOKEN_CHUNK:-8}"
STREAM_OVERLAP_LEN="${STREAM_OVERLAP_LEN:-1024}"

VAD_MODE="${VAD_MODE:-webrtc}"
VAD_DB="${VAD_DB:--55}"
VAD_FRAME_MS="${VAD_FRAME_MS:-10}"
VAD_HANGOVER_MS="${VAD_HANGOVER_MS:-200}"
VAD_WEBRTC_AGGR="${VAD_WEBRTC_AGGR:-2}"
VAD_WEBRTC_FRAME_MS="${VAD_WEBRTC_FRAME_MS:-30}"
VAD_WEBRTC_MIN_RATIO="${VAD_WEBRTC_MIN_RATIO:-0.1}"
PEAK_LIMIT="${PEAK_LIMIT:-0.99}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[genvc_run] Missing model at ${MODEL_PATH} (did you run genvc_setup_gpu.sh?)" >&2
  exit 1
fi

echo "[genvc_run] Using model=${MODEL_PATH}"
echo "[genvc_run] Streaming: chunk_ms=${STREAM_CHUNK_MS} token_chunk=${STREAM_TOKEN_CHUNK} overlap_len=${STREAM_OVERLAP_LEN}"

python -m evaluation.vc_quest.genvc_convert \
  --genvc_dir "${GENVC_DIR}" \
  --model_path "${MODEL_PATH}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --top_k "${TOP_K}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json" \
  --peak_limit "${PEAK_LIMIT}"

python -m evaluation.vc_quest.genvc_convert \
  --genvc_dir "${GENVC_DIR}" \
  --model_path "${MODEL_PATH}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --top_k "${TOP_K}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json" \
  --peak_limit "${PEAK_LIMIT}"

python -m evaluation.vc_quest.genvc_convert \
  --genvc_dir "${GENVC_DIR}" \
  --model_path "${MODEL_PATH}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --top_k "${TOP_K}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_stream.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream.meta.json" \
  --stream \
  --chunk_ms "${STREAM_CHUNK_MS}" \
  --stream_chunk_size "${STREAM_TOKEN_CHUNK}" \
  --overlap_len "${STREAM_OVERLAP_LEN}" \
  --vad_mode "${VAD_MODE}" \
  --vad_db "${VAD_DB}" \
  --vad_frame_ms "${VAD_FRAME_MS}" \
  --vad_hangover_ms "${VAD_HANGOVER_MS}" \
  --vad_webrtc_aggressiveness "${VAD_WEBRTC_AGGR}" \
  --vad_webrtc_frame_ms "${VAD_WEBRTC_FRAME_MS}" \
  --vad_webrtc_min_voiced_ratio "${VAD_WEBRTC_MIN_RATIO}" \
  --peak_limit "${PEAK_LIMIT}"

python -m evaluation.vc_quest.genvc_convert \
  --genvc_dir "${GENVC_DIR}" \
  --model_path "${MODEL_PATH}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --top_k "${TOP_K}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_stream.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream.meta.json" \
  --stream \
  --chunk_ms "${STREAM_CHUNK_MS}" \
  --stream_chunk_size "${STREAM_TOKEN_CHUNK}" \
  --overlap_len "${STREAM_OVERLAP_LEN}" \
  --vad_mode "${VAD_MODE}" \
  --vad_db "${VAD_DB}" \
  --vad_frame_ms "${VAD_FRAME_MS}" \
  --vad_hangover_ms "${VAD_HANGOVER_MS}" \
  --vad_webrtc_aggressiveness "${VAD_WEBRTC_AGGR}" \
  --vad_webrtc_frame_ms "${VAD_WEBRTC_FRAME_MS}" \
  --vad_webrtc_min_voiced_ratio "${VAD_WEBRTC_MIN_RATIO}" \
  --peak_limit "${PEAK_LIMIT}"

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

echo "[genvc_run] Wrote artifacts to ${RUN_DIR}"

