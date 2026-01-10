#!/usr/bin/env bash
set -euo pipefail

# Runs ChatterboxVC for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="chatterbox"

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/chatterbox/user_pair"
mkdir -p "${RUN_DIR}"

CHATTERBOX_DIR="${HOME}/deps/chatterbox"

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

SEED="${SEED:-0}"
CFM_TIMESTEPS="${CFM_TIMESTEPS:-10}"
WATERMARK="${WATERMARK:-true}"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-800}"
STREAM_HOP_MS="${STREAM_HOP_MS:-400}"
STREAM_FADE_MS="${STREAM_FADE_MS:-10}"
STREAM_EMIT_ALIGN="${STREAM_EMIT_ALIGN:-center}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

export HF_HOME="${HOME}/.hf_home"

echo "[chatterbox_run] chatterbox_dir=${CHATTERBOX_DIR} cfm_timesteps=${CFM_TIMESTEPS} watermark=${WATERMARK}"
echo "[chatterbox_run] Streaming: window=${STREAM_WINDOW_MS}ms hop=${STREAM_HOP_MS}ms emit_align=${STREAM_EMIT_ALIGN}"

WATERMARK_FLAG="--watermark"
if [[ "${WATERMARK}" == "0" || "${WATERMARK}" == "false" || "${WATERMARK}" == "False" ]]; then
  WATERMARK_FLAG="--no-watermark"
fi

python -m evaluation.vc_quest.chatterbox_convert \
  --chatterbox_dir "${CHATTERBOX_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --cfm_timesteps "${CFM_TIMESTEPS}" \
  ${WATERMARK_FLAG} \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json"

python -m evaluation.vc_quest.chatterbox_convert \
  --chatterbox_dir "${CHATTERBOX_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --cfm_timesteps "${CFM_TIMESTEPS}" \
  ${WATERMARK_FLAG} \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json"

python -m evaluation.vc_quest.chatterbox_convert \
  --chatterbox_dir "${CHATTERBOX_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --cfm_timesteps "${CFM_TIMESTEPS}" \
  ${WATERMARK_FLAG} \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_stream.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}" \
  --emit_align "${STREAM_EMIT_ALIGN}"

python -m evaluation.vc_quest.chatterbox_convert \
  --chatterbox_dir "${CHATTERBOX_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --cfm_timesteps "${CFM_TIMESTEPS}" \
  ${WATERMARK_FLAG} \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_stream.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}" \
  --emit_align "${STREAM_EMIT_ALIGN}"

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

echo "[chatterbox_run] Wrote artifacts to ${RUN_DIR}"
