#!/usr/bin/env bash
set -euo pipefail

# Runs QuickVC for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="quickvc"

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/quickvc/user_pair"
mkdir -p "${RUN_DIR}"

QUICKVC_DIR="${HOME}/deps/QuickVC"

# Keep this short for the first viability check; set MAX_SEC=0 for full source files.
MAX_SEC="${MAX_SEC:-10}"
SEED="${SEED:-0}"

WINDOW_MS="${WINDOW_MS:-800}"
HOP_MS="${HOP_MS:-400}"
FADE_MS="${FADE_MS:-10}"
EMIT_ALIGN="${EMIT_ALIGN:-end}"
NORM_ALIGN="${NORM_ALIGN:-end}"
PEAK_LIMIT="${PEAK_LIMIT:-0.99}"
VAD_MODE="${VAD_MODE:-off}" # off|rms|webrtc

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

CFG="${QUICKVC_DIR}/logs/quickvc/config.json"
CKPT="${QUICKVC_DIR}/logs/quickvc/quickvc.pth"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

echo "[quickvc_run] Using QuickVC repo: ${QUICKVC_DIR}"
echo "[quickvc_run] max_sec=${MAX_SEC} window_ms=${WINDOW_MS} hop_ms=${HOP_MS} emit_align=${EMIT_ALIGN} vad=${VAD_MODE}"

python -m evaluation.vc_quest.quickvc_convert \
  --quickvc_dir "${QUICKVC_DIR}" \
  --config "${CFG}" \
  --ckpt "${CKPT}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --max_sec "${MAX_SEC}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json" \
  --peak_limit "${PEAK_LIMIT}"

python -m evaluation.vc_quest.quickvc_convert \
  --quickvc_dir "${QUICKVC_DIR}" \
  --config "${CFG}" \
  --ckpt "${CKPT}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --max_sec "${MAX_SEC}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json" \
  --peak_limit "${PEAK_LIMIT}"

python -m evaluation.vc_quest.quickvc_convert \
  --quickvc_dir "${QUICKVC_DIR}" \
  --config "${CFG}" \
  --ckpt "${CKPT}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --max_sec "${MAX_SEC}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_stream.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream.meta.json" \
  --stream \
  --window_ms "${WINDOW_MS}" \
  --hop_ms "${HOP_MS}" \
  --fade_ms "${FADE_MS}" \
  --emit_align "${EMIT_ALIGN}" \
  --normalize_align "${NORM_ALIGN}" \
  --vad_mode "${VAD_MODE}" \
  --peak_limit "${PEAK_LIMIT}"

python -m evaluation.vc_quest.quickvc_convert \
  --quickvc_dir "${QUICKVC_DIR}" \
  --config "${CFG}" \
  --ckpt "${CKPT}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --max_sec "${MAX_SEC}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_stream.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream.meta.json" \
  --stream \
  --window_ms "${WINDOW_MS}" \
  --hop_ms "${HOP_MS}" \
  --fade_ms "${FADE_MS}" \
  --emit_align "${EMIT_ALIGN}" \
  --normalize_align "${NORM_ALIGN}" \
  --vad_mode "${VAD_MODE}" \
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

echo "[quickvc_run] Wrote artifacts to ${RUN_DIR}"

