#!/usr/bin/env bash
set -euo pipefail

# Runs YingMusic-SVC for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="ymsvc"

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/yingmusic_svc/user_pair"
mkdir -p "${RUN_DIR}"

YMSVC_DIR="${HOME}/deps/YingMusic-SVC"

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

CHECKPOINT_REPO="${CHECKPOINT_REPO:-GiantAILab/YingMusic-SVC}"
CHECKPOINT_FILENAME="${CHECKPOINT_FILENAME:-YingMusic-SVC-full.pt}"

DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
LENGTH_ADJUST="${LENGTH_ADJUST:-1.0}"
FP16="${FP16:-true}"
SEED="${SEED:-0}"
REF_MAX_SEC="${REF_MAX_SEC:-10.0}"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-600}"
STREAM_HOP_MS="${STREAM_HOP_MS:-300}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

echo "[ymsvc_run] ckpt=${CHECKPOINT_REPO}/${CHECKPOINT_FILENAME} steps=${DIFFUSION_STEPS} cfg=${INFERENCE_CFG_RATE} fp16=${FP16}"
echo "[ymsvc_run] Streaming: window=${STREAM_WINDOW_MS}ms hop=${STREAM_HOP_MS}ms seed=${SEED}"

python -m evaluation.vc_quest.yingmusic_svc_convert \
  --ymsvc_dir "${YMSVC_DIR}" \
  --checkpoint_repo "${CHECKPOINT_REPO}" \
  --checkpoint_filename "${CHECKPOINT_FILENAME}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --fp16="${FP16}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  --inference_cfg_rate "${INFERENCE_CFG_RATE}" \
  --length_adjust "${LENGTH_ADJUST}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json"

python -m evaluation.vc_quest.yingmusic_svc_convert \
  --ymsvc_dir "${YMSVC_DIR}" \
  --checkpoint_repo "${CHECKPOINT_REPO}" \
  --checkpoint_filename "${CHECKPOINT_FILENAME}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --fp16="${FP16}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  --inference_cfg_rate "${INFERENCE_CFG_RATE}" \
  --length_adjust "${LENGTH_ADJUST}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json"

python -m evaluation.vc_quest.yingmusic_svc_convert \
  --ymsvc_dir "${YMSVC_DIR}" \
  --checkpoint_repo "${CHECKPOINT_REPO}" \
  --checkpoint_filename "${CHECKPOINT_FILENAME}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --fp16="${FP16}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  --inference_cfg_rate "${INFERENCE_CFG_RATE}" \
  --length_adjust "${LENGTH_ADJUST}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_stream.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}"

python -m evaluation.vc_quest.yingmusic_svc_convert \
  --ymsvc_dir "${YMSVC_DIR}" \
  --checkpoint_repo "${CHECKPOINT_REPO}" \
  --checkpoint_filename "${CHECKPOINT_FILENAME}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --fp16="${FP16}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  --inference_cfg_rate "${INFERENCE_CFG_RATE}" \
  --length_adjust "${LENGTH_ADJUST}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_stream.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}"

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

echo "[ymsvc_run] Wrote artifacts to ${RUN_DIR}"

