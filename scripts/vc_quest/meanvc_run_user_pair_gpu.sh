#!/usr/bin/env bash
set -euo pipefail

# Runs MeanVC for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="meanvc"

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/meanvc/user_pair"
mkdir -p "${RUN_DIR}"

MEANVC_DIR="${HOME}/deps/MeanVC"

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

SEED="${SEED:-0}"
STEPS="${STEPS:-2}"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-200}"
STREAM_HOP_MS="${STREAM_HOP_MS:-200}"
STREAM_FADE_MS="${STREAM_FADE_MS:-10}"

DEVICE="${DEVICE:-cuda:0}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

echo "[meanvc_run] Using meanvc_dir=${MEANVC_DIR} steps=${STEPS} device=${DEVICE}"
echo "[meanvc_run] Streaming: window=${STREAM_WINDOW_MS}ms hop=${STREAM_HOP_MS}ms seed=${SEED}"

python -m evaluation.vc_quest.meanvc_convert \
  --meanvc_dir "${MEANVC_DIR}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --steps "${STEPS}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json"

python -m evaluation.vc_quest.meanvc_convert \
  --meanvc_dir "${MEANVC_DIR}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --steps "${STEPS}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json"

python -m evaluation.vc_quest.meanvc_convert \
  --meanvc_dir "${MEANVC_DIR}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --steps "${STEPS}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_stream_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}"

python -m evaluation.vc_quest.meanvc_convert \
  --meanvc_dir "${MEANVC_DIR}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --steps "${STEPS}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_stream_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}.meta.json" \
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
  --deg_wav "${RUN_DIR}/v5_to_fr_stream_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}.meta.json" \
  --out_json "${RUN_DIR}/v5_to_fr_stream_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}.report.json"

python -m evaluation.vc_quest.score_outputs \
  --ref_wav "${REF_V5}" \
  --src_wav "${SRC_FR}" \
  --deg_wav "${RUN_DIR}/fr_to_v5_stream_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}.meta.json" \
  --out_json "${RUN_DIR}/fr_to_v5_stream_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}.report.json"

python -m evaluation.vc_quest.select_best --run_dir "${RUN_DIR}"

echo "[meanvc_run] Wrote artifacts to ${RUN_DIR}"

