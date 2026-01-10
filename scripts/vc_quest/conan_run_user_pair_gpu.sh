#!/usr/bin/env bash
set -euo pipefail

# Runs Conan for the user pair (offline + streaming sim),
# then scores outputs using Amphion evaluation tools.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="conan"

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/conan/user_pair"
mkdir -p "${RUN_DIR}"

CONAN_DIR="${HOME}/deps/Conan"

# Keep this short for the first viability check; set MAX_SEC=0 for full source files.
MAX_SEC="${MAX_SEC:-10}"

SEED="${SEED:-0}"

CHUNK_MS="${CHUNK_MS:-80}"
RIGHT_CONTEXT="${RIGHT_CONTEXT:-2}"
MODEL_CONTEXT_FRAMES="${MODEL_CONTEXT_FRAMES:-64}"
VOCODER_CONTEXT_FRAMES="${VOCODER_CONTEXT_FRAMES:-4}"
FADE_MS="${FADE_MS:-10}"
PEAK_LIMIT="${PEAK_LIMIT:-0.99}"

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

# Conan config uses relative `base_config:` paths, so we keep `--config` relative to Conan repo root.
CONAN_CONFIG="egs/conan_emformer.yaml"
CONAN_EXP_NAME="Conan"

# Conan's public config uses `checkpoints/emformer_test2`; override to downloaded folder name.
HP_OVERRIDE="emformer_ckpt=checkpoints/Emformer,vocoder_ckpt=checkpoints/hifigan_vc"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

echo "[conan_run] Using Conan repo: ${CONAN_DIR}"
echo "[conan_run] max_sec=${MAX_SEC} chunk_ms=${CHUNK_MS} rc=${RIGHT_CONTEXT} model_ctx_frames=${MODEL_CONTEXT_FRAMES} voc_ctx_frames=${VOCODER_CONTEXT_FRAMES}"

python -m evaluation.vc_quest.conan_convert \
  --conan_dir "${CONAN_DIR}" \
  --config "${CONAN_CONFIG}" \
  --exp_name "${CONAN_EXP_NAME}" \
  --hparams_override "${HP_OVERRIDE}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --max_sec "${MAX_SEC}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json" \
  --peak_limit "${PEAK_LIMIT}"

python -m evaluation.vc_quest.conan_convert \
  --conan_dir "${CONAN_DIR}" \
  --config "${CONAN_CONFIG}" \
  --exp_name "${CONAN_EXP_NAME}" \
  --hparams_override "${HP_OVERRIDE}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --max_sec "${MAX_SEC}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json" \
  --peak_limit "${PEAK_LIMIT}"

python -m evaluation.vc_quest.conan_convert \
  --conan_dir "${CONAN_DIR}" \
  --config "${CONAN_CONFIG}" \
  --exp_name "${CONAN_EXP_NAME}" \
  --hparams_override "${HP_OVERRIDE}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --max_sec "${MAX_SEC}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_stream.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_stream.meta.json" \
  --stream \
  --chunk_ms "${CHUNK_MS}" \
  --right_context "${RIGHT_CONTEXT}" \
  --model_context_frames "${MODEL_CONTEXT_FRAMES}" \
  --vocoder_context_frames "${VOCODER_CONTEXT_FRAMES}" \
  --fade_ms "${FADE_MS}" \
  --peak_limit "${PEAK_LIMIT}"

python -m evaluation.vc_quest.conan_convert \
  --conan_dir "${CONAN_DIR}" \
  --config "${CONAN_CONFIG}" \
  --exp_name "${CONAN_EXP_NAME}" \
  --hparams_override "${HP_OVERRIDE}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --max_sec "${MAX_SEC}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_stream.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_stream.meta.json" \
  --stream \
  --chunk_ms "${CHUNK_MS}" \
  --right_context "${RIGHT_CONTEXT}" \
  --model_context_frames "${MODEL_CONTEXT_FRAMES}" \
  --vocoder_context_frames "${VOCODER_CONTEXT_FRAMES}" \
  --fade_ms "${FADE_MS}" \
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

echo "[conan_run] Wrote artifacts to ${RUN_DIR}"

