#!/usr/bin/env bash
set -euo pipefail

# Runs FragmentVC for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="fragmentvc"

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/fragmentvc/user_pair"
mkdir -p "${RUN_DIR}"

FRAGMENTVC_DIR="${HOME}/deps/FragmentVC"
CKPT_PATH="${FRAGMENTVC_DIR}/pretrained/fragmentvc.pt"
VOCODER_PATH="${FRAGMENTVC_DIR}/pretrained/vocoder.pt"

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

WAV2VEC_MODEL="${WAV2VEC_MODEL:-facebook/wav2vec2-base}"
SEED="${SEED:-0}"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-600}"
STREAM_HOP_MS="${STREAM_HOP_MS:-300}"
STREAM_FADE_MS="${STREAM_FADE_MS:-10}"

VAD_MODE="${VAD_MODE:-webrtc}"
VAD_DB="${VAD_DB:--55}"
VAD_FRAME_MS="${VAD_FRAME_MS:-10}"
VAD_HANGOVER_MS="${VAD_HANGOVER_MS:-200}"
PEAK_LIMIT="${PEAK_LIMIT:-0.99}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

echo "[fragmentvc_run] Using ckpt=${CKPT_PATH} vocoder=${VOCODER_PATH}"
echo "[fragmentvc_run] wav2vec_model=${WAV2VEC_MODEL}"
echo "[fragmentvc_run] Streaming: window=${STREAM_WINDOW_MS}ms hop=${STREAM_HOP_MS}ms"

python -m evaluation.vc_quest.fragmentvc_convert \
  --ckpt_path "${CKPT_PATH}" \
  --vocoder_path "${VOCODER_PATH}" \
  --wav2vec_model "${WAV2VEC_MODEL}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json"

python -m evaluation.vc_quest.fragmentvc_convert \
  --ckpt_path "${CKPT_PATH}" \
  --vocoder_path "${VOCODER_PATH}" \
  --wav2vec_model "${WAV2VEC_MODEL}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json"

python -m evaluation.vc_quest.fragmentvc_convert \
  --ckpt_path "${CKPT_PATH}" \
  --vocoder_path "${VOCODER_PATH}" \
  --wav2vec_model "${WAV2VEC_MODEL}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_stream.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}" \
  --emit_align center \
  --vad_mode "${VAD_MODE}" \
  --vad_db "${VAD_DB}" \
  --vad_frame_ms "${VAD_FRAME_MS}" \
  --vad_hangover_ms "${VAD_HANGOVER_MS}" \
  --peak_limit "${PEAK_LIMIT}"

python -m evaluation.vc_quest.fragmentvc_convert \
  --ckpt_path "${CKPT_PATH}" \
  --vocoder_path "${VOCODER_PATH}" \
  --wav2vec_model "${WAV2VEC_MODEL}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_stream.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}" \
  --emit_align center \
  --vad_mode "${VAD_MODE}" \
  --vad_db "${VAD_DB}" \
  --vad_frame_ms "${VAD_FRAME_MS}" \
  --vad_hangover_ms "${VAD_HANGOVER_MS}" \
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

echo "[fragmentvc_run] Wrote artifacts to ${RUN_DIR}"

