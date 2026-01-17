#!/usr/bin/env bash
set -euo pipefail

# Runs StreamVoiceAnon over the deterministic French playlist, then scores the outputs.

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge3}"
if [[ ! -f "${MINIFORGE_ROOT}/etc/profile.d/conda.sh" ]]; then
  if command -v conda >/dev/null 2>&1; then
    MINIFORGE_ROOT="$(conda info --base)"
  fi
fi

CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[streamvoiceanon_run_fleurs_fr_playlist] Missing conda.sh at ${CONDA_SH}. Set MINIFORGE_ROOT or install conda." >&2
  exit 1
fi

ENV_NAME="${ENV_NAME:-streamvoiceanon}"

cd "$(dirname "$0")/../.."

PLAYLIST_DIR="${PLAYLIST_DIR:-data/vc_quest_playlists/fleurs_fr_fr_dev_v1}"
MANIFEST="${MANIFEST:-${PLAYLIST_DIR}/manifest.json}"

RUN_NAME="${RUN_NAME:-streamvoiceanon_d2_c1_e128_d64}"
RUN_DIR="runs/vc_quest/playlists/fleurs_fr_fr/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

STREAMVOICEANON_DIR="${STREAMVOICEANON_DIR:-${HOME}/deps/StreamVoiceAnon}"

MODEL_DEVICE="${MODEL_DEVICE:-cuda:0}"
SEED="${SEED:-0}"

ENCODE_WINDOW_FRAMES="${ENCODE_WINDOW_FRAMES:-128}"
DECODE_WINDOW_FRAMES="${DECODE_WINDOW_FRAMES:-64}"
MAX_PROMPT_FRAMES="${MAX_PROMPT_FRAMES:-256}"
MAX_SEQ_FRAMES="${MAX_SEQ_FRAMES:-768}"
BUFFER_FRAMES="${BUFFER_FRAMES:-32}"
DECODE_CHUNK_FRAMES="${DECODE_CHUNK_FRAMES:-1}"
DELAY_FRAMES="${DELAY_FRAMES:-2}"

COMPILE_AR="${COMPILE_AR:-0}"
COMPILE_ENCODER="${COMPILE_ENCODER:-0}"
COMPILE_DECODER="${COMPILE_DECODER:-0}"

FADE_MS="${FADE_MS:-10}"
GAIN_MODE="${GAIN_MODE:-off}"
GAIN_TARGET_DELTA_DB="${GAIN_TARGET_DELTA_DB:-10.0}"
GAIN_MAX_BOOST_DB="${GAIN_MAX_BOOST_DB:-18.0}"
GAIN_SMOOTHING="${GAIN_SMOOTHING:-0.0}"
MASK_MODE="${MASK_MODE:-off}"
MASK_DB="${MASK_DB:--50.0}"
MASK_FRAME_MS="${MASK_FRAME_MS:-10.0}"
MASK_SMOOTH_MS="${MASK_SMOOTH_MS:-10.0}"
PEAK_LIMIT="${PEAK_LIMIT:-0.99}"

MAX_PAIRS="${MAX_PAIRS:-0}"
WER_MODE="${WER_MODE:-audio_ref}"

# 1) Ensure playlist exists (build in current python env; uses HF downloads).
if [[ ! -f "${MANIFEST}" ]]; then
  scripts/vc_quest/fleurs_fr_build_playlist.sh
fi

# 2) Convert in the StreamVoiceAnon conda env.
source "${CONDA_SH}"
conda activate "${ENV_NAME}"

EXTRA_ARGS=()
if [[ "${COMPILE_AR}" == "1" ]]; then
  EXTRA_ARGS+=(--compile_ar)
else
  EXTRA_ARGS+=(--no-compile_ar)
fi
if [[ "${COMPILE_ENCODER}" == "1" ]]; then
  EXTRA_ARGS+=(--compile_encoder)
else
  EXTRA_ARGS+=(--no-compile_encoder)
fi
if [[ "${COMPILE_DECODER}" == "1" ]]; then
  EXTRA_ARGS+=(--compile_decoder)
else
  EXTRA_ARGS+=(--no-compile_decoder)
fi

python -m evaluation.vc_quest.streamvoiceanon_playlist_convert \
  --manifest "${MANIFEST}" \
  --out_dir "${RUN_DIR}" \
  --streamvoiceanon_dir "${STREAMVOICEANON_DIR}" \
  --device "${MODEL_DEVICE}" \
  --seed "${SEED}" \
  --encode_window_frames "${ENCODE_WINDOW_FRAMES}" \
  --decode_window_frames "${DECODE_WINDOW_FRAMES}" \
  --max_prompt_frames "${MAX_PROMPT_FRAMES}" \
  --max_seq_frames "${MAX_SEQ_FRAMES}" \
  --buffer_frames "${BUFFER_FRAMES}" \
  --decode_chunk_frames "${DECODE_CHUNK_FRAMES}" \
  --delay_frames "${DELAY_FRAMES}" \
  --fade_ms "${FADE_MS}" \
  --gain_mode "${GAIN_MODE}" \
  --gain_target_delta_db "${GAIN_TARGET_DELTA_DB}" \
  --gain_max_boost_db "${GAIN_MAX_BOOST_DB}" \
  --gain_smoothing "${GAIN_SMOOTHING}" \
  --mask_mode "${MASK_MODE}" \
  --mask_db "${MASK_DB}" \
  --mask_frame_ms "${MASK_FRAME_MS}" \
  --mask_smooth_ms "${MASK_SMOOTH_MS}" \
  --peak_limit "${PEAK_LIMIT}" \
  --max_pairs "${MAX_PAIRS}" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

conda deactivate || true

# 3) Score in Amphion's eval venv.
source .venv/bin/activate

SCORE_ARGS=()
if [[ "${PITCH_METRICS:-0}" == "1" ]]; then
  SCORE_ARGS+=(--pitch_metrics)
fi

python -m evaluation.vc_quest.score_playlist \
  --manifest "${MANIFEST}" \
  --run_dir "${RUN_DIR}" \
  --out_json "${RUN_DIR}/summary.json" \
  --whisper_model base \
  --whisper_language fr \
  --wer_mode "${WER_MODE}" \
  "${SCORE_ARGS[@]+"${SCORE_ARGS[@]}"}"

echo "[streamvoiceanon_run_fleurs_fr_playlist] Wrote ${RUN_DIR}/summary.json"

