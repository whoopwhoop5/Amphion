#!/usr/bin/env bash
set -euo pipefail

# Runs SaMoye-SVC for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="samoye"

cd "$(dirname "$0")/../.."

RUN_TAG="${RUN_TAG:-user_pair}"
RUN_DIR="runs/vc_quest/samoye_svc/${RUN_TAG}"
mkdir -p "${RUN_DIR}"

SAMOYE_DIR="${HOME}/deps/SaMoye-SVC/SaMoye-Model"

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

FP16="${FP16:-true}"
SEED="${SEED:-0}"
REF_MAX_SEC="${REF_MAX_SEC:-10.0}"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-600}"
STREAM_HOP_MS="${STREAM_HOP_MS:-300}"
STREAM_CROSSFADE_MS="${STREAM_CROSSFADE_MS:-10}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

FP16_FLAG="--fp16"
if [[ "${FP16}" == "false" ]]; then
  FP16_FLAG="--no-fp16"
fi

echo "[samoye_run] fp16=${FP16} seed=${SEED} ref_max_sec=${REF_MAX_SEC}"
echo "[samoye_run] Streaming: window=${STREAM_WINDOW_MS}ms hop=${STREAM_HOP_MS}ms crossfade=${STREAM_CROSSFADE_MS}ms"

python -m evaluation.vc_quest.samoye_svc_convert \
  --samoye_dir "${SAMOYE_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  ${FP16_FLAG} \
  --reference_max_sec "${REF_MAX_SEC}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json"

python -m evaluation.vc_quest.samoye_svc_convert \
  --samoye_dir "${SAMOYE_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  ${FP16_FLAG} \
  --reference_max_sec "${REF_MAX_SEC}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json"

python -m evaluation.vc_quest.samoye_svc_convert \
  --samoye_dir "${SAMOYE_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  ${FP16_FLAG} \
  --reference_max_sec "${REF_MAX_SEC}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_stream.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --crossfade_ms "${STREAM_CROSSFADE_MS}"

python -m evaluation.vc_quest.samoye_svc_convert \
  --samoye_dir "${SAMOYE_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  ${FP16_FLAG} \
  --reference_max_sec "${REF_MAX_SEC}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_stream.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --crossfade_ms "${STREAM_CROSSFADE_MS}"

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

echo "[samoye_run] Wrote artifacts to ${RUN_DIR}"

