#!/usr/bin/env bash
set -euo pipefail

# Runs TinyVC for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="tinyvc"

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/tinyvc/user_pair"
mkdir -p "${RUN_DIR}"

TINYVC_DIR="${HOME}/deps/tinyvc"

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

SEED="${SEED:-0}"
PITCH_SHIFT="${PITCH_SHIFT:-0.0}"
F0_ESTIMATION="${F0_ESTIMATION:-harvest}"

BLOCK_SIZE="${BLOCK_SIZE:-1920}" # 1920 @24kHz ~= 80ms
EXTRA_SIZE="${EXTRA_SIZE:-0}"
USE_PHASE_VOCODER="${USE_PHASE_VOCODER:-false}"

VAD_MODE="${VAD_MODE:-rms}"
VAD_DB="${VAD_DB:--55}"
VAD_FRAME_MS="${VAD_FRAME_MS:-10}"
VAD_HANGOVER_MS="${VAD_HANGOVER_MS:-200}"
PEAK_LIMIT="${PEAK_LIMIT:-0.99}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

echo "[tinyvc_run] Using tinyvc_dir=${TINYVC_DIR}"
echo "[tinyvc_run] pitch_shift=${PITCH_SHIFT} f0_estimation=${F0_ESTIMATION}"
echo "[tinyvc_run] Streaming: block_size=${BLOCK_SIZE} extra_size=${EXTRA_SIZE} vad_mode=${VAD_MODE}"

PHASE_VOCODER_FLAG="--no-use_phase_vocoder"
if [[ "${USE_PHASE_VOCODER}" == "true" ]]; then
  PHASE_VOCODER_FLAG="--use_phase_vocoder"
fi

python -m evaluation.vc_quest.tinyvc_convert \
  --tinyvc_dir "${TINYVC_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --pitch_shift "${PITCH_SHIFT}" \
  --f0_estimation "${F0_ESTIMATION}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json"

python -m evaluation.vc_quest.tinyvc_convert \
  --tinyvc_dir "${TINYVC_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --pitch_shift "${PITCH_SHIFT}" \
  --f0_estimation "${F0_ESTIMATION}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json"

python -m evaluation.vc_quest.tinyvc_convert \
  --tinyvc_dir "${TINYVC_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --pitch_shift "${PITCH_SHIFT}" \
  --f0_estimation "${F0_ESTIMATION}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_stream.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream.meta.json" \
  --stream \
  --block_size "${BLOCK_SIZE}" \
  --extra_size "${EXTRA_SIZE}" \
  ${PHASE_VOCODER_FLAG} \
  --vad_mode "${VAD_MODE}" \
  --vad_db "${VAD_DB}" \
  --vad_frame_ms "${VAD_FRAME_MS}" \
  --vad_hangover_ms "${VAD_HANGOVER_MS}" \
  --peak_limit "${PEAK_LIMIT}"

python -m evaluation.vc_quest.tinyvc_convert \
  --tinyvc_dir "${TINYVC_DIR}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --pitch_shift "${PITCH_SHIFT}" \
  --f0_estimation "${F0_ESTIMATION}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_stream.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream.meta.json" \
  --stream \
  --block_size "${BLOCK_SIZE}" \
  --extra_size "${EXTRA_SIZE}" \
  ${PHASE_VOCODER_FLAG} \
  --vad_mode "${VAD_MODE}" \
  --vad_db "${VAD_DB}" \
  --vad_frame_ms "${VAD_FRAME_MS}" \
  --vad_hangover_ms "${VAD_HANGOVER_MS}" \
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

echo "[tinyvc_run] Wrote artifacts to ${RUN_DIR}"
