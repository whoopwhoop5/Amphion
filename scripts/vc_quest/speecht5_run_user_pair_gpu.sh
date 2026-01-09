#!/usr/bin/env bash
set -euo pipefail

# Runs SpeechT5 VC for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="speecht5"

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/speecht5/user_pair"
mkdir -p "${RUN_DIR}"

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

VC_MODEL="${VC_MODEL:-microsoft/speecht5_vc}"
VOCODER_MODEL="${VOCODER_MODEL:-microsoft/speecht5_hifigan}"
SPEAKER_MODEL="${SPEAKER_MODEL:-microsoft/wavlm-base-plus-sv}"
SEED="${SEED:-0}"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-600}"
STREAM_HOP_MS="${STREAM_HOP_MS:-300}"
STREAM_FADE_MS="${STREAM_FADE_MS:-10}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

echo "[speecht5_run] Using vc_model=${VC_MODEL} vocoder=${VOCODER_MODEL} speaker_model=${SPEAKER_MODEL}"
echo "[speecht5_run] Streaming: window=${STREAM_WINDOW_MS}ms hop=${STREAM_HOP_MS}ms seed=${SEED}"

python -m evaluation.vc_quest.speecht5_convert \
  --vc_model "${VC_MODEL}" \
  --vocoder_model "${VOCODER_MODEL}" \
  --speaker_model "${SPEAKER_MODEL}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json"

python -m evaluation.vc_quest.speecht5_convert \
  --vc_model "${VC_MODEL}" \
  --vocoder_model "${VOCODER_MODEL}" \
  --speaker_model "${SPEAKER_MODEL}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json"

python -m evaluation.vc_quest.speecht5_convert \
  --vc_model "${VC_MODEL}" \
  --vocoder_model "${VOCODER_MODEL}" \
  --speaker_model "${SPEAKER_MODEL}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_stream.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}"

python -m evaluation.vc_quest.speecht5_convert \
  --vc_model "${VC_MODEL}" \
  --vocoder_model "${VOCODER_MODEL}" \
  --speaker_model "${SPEAKER_MODEL}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_stream.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}"

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

echo "[speecht5_run] Wrote artifacts to ${RUN_DIR}"

