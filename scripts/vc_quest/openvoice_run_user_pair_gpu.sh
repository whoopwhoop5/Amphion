#!/usr/bin/env bash
set -euo pipefail

# Runs OpenVoice tone-color VC for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="openvoice"

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/openvoice/user_pair"
mkdir -p "${RUN_DIR}"

CKPT_DIR="${HOME}/deps/OpenVoice/checkpoints_v2/converter"

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-600}"
STREAM_HOP_MS="${STREAM_HOP_MS:-600}"
STREAM_FADE_MS="${STREAM_FADE_MS:-10}"
STREAM_TAU="${STREAM_TAU:-0.3}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

echo "[openvoice_run] Using ckpt_dir=${CKPT_DIR}"
echo "[openvoice_run] Streaming: window=${STREAM_WINDOW_MS}ms hop=${STREAM_HOP_MS}ms tau=${STREAM_TAU}"

python -m evaluation.vc_quest.openvoice_convert \
  --ckpt_dir "${CKPT_DIR}" \
  --device cuda:0 \
  --tau "${STREAM_TAU}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json"

python -m evaluation.vc_quest.openvoice_convert \
  --ckpt_dir "${CKPT_DIR}" \
  --device cuda:0 \
  --tau "${STREAM_TAU}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json"

python -m evaluation.vc_quest.openvoice_convert \
  --ckpt_dir "${CKPT_DIR}" \
  --device cuda:0 \
  --tau "${STREAM_TAU}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_stream.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}"

python -m evaluation.vc_quest.openvoice_convert \
  --ckpt_dir "${CKPT_DIR}" \
  --device cuda:0 \
  --tau "${STREAM_TAU}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_stream.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}"

deactivate || true
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

echo "[openvoice_run] Wrote artifacts to ${RUN_DIR}"

