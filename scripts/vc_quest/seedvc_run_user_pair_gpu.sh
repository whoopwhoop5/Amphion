#!/usr/bin/env bash
set -euo pipefail

# Runs Seed-VC (xlsr-tiny realtime model) for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="seedvc"

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/seedvc/user_pair"
mkdir -p "${RUN_DIR}"

SEEDVC_DIR="${HOME}/deps/seed-vc"

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

SEED="${SEED:-0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-10}"
INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
MAX_PROMPT_SEC="${MAX_PROMPT_SEC:-3.0}"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-300}"
STREAM_HOP_MS="${STREAM_HOP_MS:-300}"
CROSSFADE_MS="${CROSSFADE_MS:-40}"
EXTRA_TIME_CE_MS="${EXTRA_TIME_CE_MS:-2500}"
EXTRA_TIME_MS="${EXTRA_TIME_MS:-500}"
EXTRA_TIME_RIGHT_MS="${EXTRA_TIME_RIGHT_MS:-20}"

VAD_MODE="${VAD_MODE:-rms}"
VAD_DB="${VAD_DB:--55.0}"
PEAK_LIMIT="${PEAK_LIMIT:-0.99}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

echo "[seedvc_run] Using seedvc_dir=${SEEDVC_DIR} steps=${DIFFUSION_STEPS} cfg=${INFERENCE_CFG_RATE} seed=${SEED}"
echo "[seedvc_run] Stream: window=${STREAM_WINDOW_MS}ms crossfade=${CROSSFADE_MS}ms extra_ce=${EXTRA_TIME_CE_MS}ms extra=${EXTRA_TIME_MS}ms right=${EXTRA_TIME_RIGHT_MS}ms"

python -m evaluation.vc_quest.seedvc_convert \
  --seedvc_dir "${SEEDVC_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --fp16 \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  --inference_cfg_rate "${INFERENCE_CFG_RATE}" \
  --max_prompt_length_sec "${MAX_PROMPT_SEC}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json"

python -m evaluation.vc_quest.seedvc_convert \
  --seedvc_dir "${SEEDVC_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --fp16 \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  --inference_cfg_rate "${INFERENCE_CFG_RATE}" \
  --max_prompt_length_sec "${MAX_PROMPT_SEC}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json"

python -m evaluation.vc_quest.seedvc_convert \
  --seedvc_dir "${SEEDVC_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --fp16 \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  --inference_cfg_rate "${INFERENCE_CFG_RATE}" \
  --max_prompt_length_sec "${MAX_PROMPT_SEC}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_stream_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --crossfade_ms "${CROSSFADE_MS}" \
  --extra_time_ce_ms "${EXTRA_TIME_CE_MS}" \
  --extra_time_ms "${EXTRA_TIME_MS}" \
  --extra_time_right_ms "${EXTRA_TIME_RIGHT_MS}" \
  --vad_mode "${VAD_MODE}" \
  --vad_db "${VAD_DB}" \
  --peak_limit "${PEAK_LIMIT}"

python -m evaluation.vc_quest.seedvc_convert \
  --seedvc_dir "${SEEDVC_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --fp16 \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  --inference_cfg_rate "${INFERENCE_CFG_RATE}" \
  --max_prompt_length_sec "${MAX_PROMPT_SEC}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_stream_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}.meta.json" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --crossfade_ms "${CROSSFADE_MS}" \
  --extra_time_ce_ms "${EXTRA_TIME_CE_MS}" \
  --extra_time_ms "${EXTRA_TIME_MS}" \
  --extra_time_right_ms "${EXTRA_TIME_RIGHT_MS}" \
  --vad_mode "${VAD_MODE}" \
  --vad_db "${VAD_DB}" \
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

echo "[seedvc_run] Wrote artifacts to ${RUN_DIR}"

