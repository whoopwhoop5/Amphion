#!/usr/bin/env bash
set -euo pipefail

# Runs FACodec VC for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/facodec/user_pair"
mkdir -p "${RUN_DIR}"

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

HF_REPO="${HF_REPO:-amphion/naturalspeech3_facodec}"
SEED="${SEED:-0}"
REF_MAX_SEC="${REF_MAX_SEC:-10.0}"
USE_RESIDUAL="${USE_RESIDUAL:-false}"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-600}"
STREAM_HOP_MS="${STREAM_HOP_MS:-300}"
STREAM_FADE_MS="${STREAM_FADE_MS:-10}"

source .venv/bin/activate

# Keep HF token + caches off /workspace to avoid disk-full failures (and to use the login token).
export HF_HOME="/root/.hf_home"
mkdir -p "${HF_HOME}"

echo "[facodec_run] hf_repo=${HF_REPO} ref_max_sec=${REF_MAX_SEC} use_residual=${USE_RESIDUAL} seed=${SEED}"
echo "[facodec_run] Streaming: window=${STREAM_WINDOW_MS}ms hop=${STREAM_HOP_MS}ms fade=${STREAM_FADE_MS}ms"

python -m evaluation.vc_quest.facodec_convert \
  --hf_repo "${HF_REPO}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --use_residual "${USE_RESIDUAL}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json"

python -m evaluation.vc_quest.facodec_convert \
  --hf_repo "${HF_REPO}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --use_residual "${USE_RESIDUAL}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json"

python -m evaluation.vc_quest.facodec_convert \
  --hf_repo "${HF_REPO}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --use_residual "${USE_RESIDUAL}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_stream.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}"

python -m evaluation.vc_quest.facodec_convert \
  --hf_repo "${HF_REPO}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref_max_sec "${REF_MAX_SEC}" \
  --use_residual "${USE_RESIDUAL}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_stream.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}"

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

echo "[facodec_run] Wrote artifacts to ${RUN_DIR}"

