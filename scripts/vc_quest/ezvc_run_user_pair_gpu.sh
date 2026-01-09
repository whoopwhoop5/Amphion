#!/usr/bin/env bash
set -euo pipefail

# Runs EZ-VC for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="ezvc"

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/ezvc/user_pair"
mkdir -p "${RUN_DIR}"

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

MODEL_REPO="${MODEL_REPO:-SPRINGLab/EZ-VC}"
VOCODER_NAME="${VOCODER_NAME:-bigvgan}"
SEED="${SEED:-0}"
REF_MAX_SEC="${REF_MAX_SEC:-6.0}"

NFE_STEP="${NFE_STEP:-12}"
CFG_STRENGTH="${CFG_STRENGTH:-2.0}"
SWAY_SAMPLING_COEF="${SWAY_SAMPLING_COEF:--1.0}"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-600}"
STREAM_HOP_MS="${STREAM_HOP_MS:-300}"
STREAM_FADE_MS="${STREAM_FADE_MS:-10}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

# Keep HF token + caches off /workspace to avoid disk-full failures (and to use the login token).
export HF_HOME="/root/.hf_home"
mkdir -p "${HF_HOME}"

echo "[ezvc_run] model_repo=${MODEL_REPO} vocoder=${VOCODER_NAME} nfe=${NFE_STEP} cfg=${CFG_STRENGTH} sway=${SWAY_SAMPLING_COEF}"
echo "[ezvc_run] Streaming: window=${STREAM_WINDOW_MS}ms hop=${STREAM_HOP_MS}ms fade=${STREAM_FADE_MS}ms seed=${SEED}"

python -m evaluation.vc_quest.ezvc_convert \
  --model_repo "${MODEL_REPO}" \
  --vocoder_name "${VOCODER_NAME}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --nfe_step "${NFE_STEP}" \
  --cfg_strength "${CFG_STRENGTH}" \
  --sway_sampling_coef "${SWAY_SAMPLING_COEF}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json"

python -m evaluation.vc_quest.ezvc_convert \
  --model_repo "${MODEL_REPO}" \
  --vocoder_name "${VOCODER_NAME}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --nfe_step "${NFE_STEP}" \
  --cfg_strength "${CFG_STRENGTH}" \
  --sway_sampling_coef "${SWAY_SAMPLING_COEF}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json"

python -m evaluation.vc_quest.ezvc_convert \
  --model_repo "${MODEL_REPO}" \
  --vocoder_name "${VOCODER_NAME}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --nfe_step "${NFE_STEP}" \
  --cfg_strength "${CFG_STRENGTH}" \
  --sway_sampling_coef "${SWAY_SAMPLING_COEF}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_stream.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}"

python -m evaluation.vc_quest.ezvc_convert \
  --model_repo "${MODEL_REPO}" \
  --vocoder_name "${VOCODER_NAME}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --nfe_step "${NFE_STEP}" \
  --cfg_strength "${CFG_STRENGTH}" \
  --sway_sampling_coef "${SWAY_SAMPLING_COEF}" \
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

echo "[ezvc_run] Wrote artifacts to ${RUN_DIR}"
