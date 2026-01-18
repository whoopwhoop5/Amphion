#!/usr/bin/env bash
set -euo pipefail

# Runs Seed-VC over the deterministic French playlist, then scores the outputs.

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge3}"
if [[ ! -f "${MINIFORGE_ROOT}/etc/profile.d/conda.sh" ]]; then
  if command -v conda >/dev/null 2>&1; then
    MINIFORGE_ROOT="$(conda info --base)"
  fi
fi

CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[seedvc_run_fleurs_fr_playlist] Missing conda.sh at ${CONDA_SH}. Set MINIFORGE_ROOT or install conda." >&2
  exit 1
fi

ENV_NAME="${ENV_NAME:-seedvc}"

cd "$(dirname "$0")/../.."

PLAYLIST_DIR="${PLAYLIST_DIR:-data/vc_quest_playlists/fleurs_fr_fr_dev_v1}"
MANIFEST="${MANIFEST:-${PLAYLIST_DIR}/manifest.json}"

RUN_NAME="${RUN_NAME:-seedvc_xlsr_tiny_w300_h300}"
RUN_DIR="runs/vc_quest/playlists/fleurs_fr_fr/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

SEEDVC_DIR="${SEEDVC_DIR:-${HOME}/deps/seed-vc}"
MODEL_DEVICE="${MODEL_DEVICE:-cuda:0}"
SEED="${SEED:-0}"
FP16="${FP16:-1}"

HF_REPO="${HF_REPO:-Plachta/Seed-VC}"
HF_CHECKPOINT_NAME="${HF_CHECKPOINT_NAME:-DiT_uvit_tat_xlsr_ema.pth}"
HF_CONFIG_NAME="${HF_CONFIG_NAME:-config_dit_mel_seed_uvit_xlsr_tiny.yml}"

DIFFUSION_STEPS="${DIFFUSION_STEPS:-10}"
INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
MAX_PROMPT_SEC="${MAX_PROMPT_SEC:-3.0}"
LENGTH_ADJUST="${LENGTH_ADJUST:-1.0}"

STREAM="${STREAM:-1}"
STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-300}"
STREAM_HOP_MS="${STREAM_HOP_MS:-300}"
EMIT_ALIGN="${EMIT_ALIGN:-center}"
CROSSFADE_MS="${CROSSFADE_MS:-40}"
EXTRA_TIME_CE_MS="${EXTRA_TIME_CE_MS:-2500}"
EXTRA_TIME_MS="${EXTRA_TIME_MS:-500}"
EXTRA_TIME_RIGHT_MS="${EXTRA_TIME_RIGHT_MS:-20}"
DROP_WARMUP_HOPS="${DROP_WARMUP_HOPS:-1}"

VAD_MODE="${VAD_MODE:-rms}"
VAD_DB="${VAD_DB:--55.0}"
VAD_FRAME_MS="${VAD_FRAME_MS:-10.0}"
VAD_HANGOVER_MS="${VAD_HANGOVER_MS:-200.0}"
VAD_WEBRTC_FRAME_MS="${VAD_WEBRTC_FRAME_MS:-30}"
VAD_WEBRTC_AGGRESSIVENESS="${VAD_WEBRTC_AGGRESSIVENESS:-2}"
VAD_WEBRTC_MIN_VOICED_RATIO="${VAD_WEBRTC_MIN_VOICED_RATIO:-0.1}"

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

# Vast images often set HF_HOME=/workspace/.hf_home (20GB disk, frequently full).
# Override to a larger path on the container overlay FS.
if [[ -z "${HF_HOME:-}" || "${HF_HOME}" == /workspace/* ]]; then
  export HF_HOME="${HOME}/.hf_home"
fi
mkdir -p "${HF_HOME}"

# 1) Ensure playlist exists (build in current python env; uses HF downloads).
if [[ ! -f "${MANIFEST}" ]]; then
  scripts/vc_quest/fleurs_fr_build_playlist.sh
fi

# 2) Convert in the Seed-VC conda env.
source "${CONDA_SH}"
conda activate "${ENV_NAME}"

FP16_ARGS=()
if [[ "${FP16}" == "1" ]]; then
  FP16_ARGS+=(--fp16)
else
  FP16_ARGS+=(--no-fp16)
fi

STREAM_ARGS=()
if [[ "${STREAM}" == "1" ]]; then
  STREAM_ARGS+=(--stream)
  STREAM_ARGS+=(--window_ms "${STREAM_WINDOW_MS}")
  STREAM_ARGS+=(--hop_ms "${STREAM_HOP_MS}")
  STREAM_ARGS+=(--emit_align "${EMIT_ALIGN}")
  STREAM_ARGS+=(--crossfade_ms "${CROSSFADE_MS}")
  STREAM_ARGS+=(--extra_time_ce_ms "${EXTRA_TIME_CE_MS}")
  STREAM_ARGS+=(--extra_time_ms "${EXTRA_TIME_MS}")
  STREAM_ARGS+=(--extra_time_right_ms "${EXTRA_TIME_RIGHT_MS}")

  if [[ "${DROP_WARMUP_HOPS}" == "1" ]]; then
    STREAM_ARGS+=(--drop_warmup_hops)
  else
    STREAM_ARGS+=(--no-drop_warmup_hops)
  fi

  STREAM_ARGS+=(--vad_mode "${VAD_MODE}")
  STREAM_ARGS+=(--vad_db "${VAD_DB}")
  STREAM_ARGS+=(--vad_frame_ms "${VAD_FRAME_MS}")
  STREAM_ARGS+=(--vad_hangover_ms "${VAD_HANGOVER_MS}")
  STREAM_ARGS+=(--vad_webrtc_frame_ms "${VAD_WEBRTC_FRAME_MS}")
  STREAM_ARGS+=(--vad_webrtc_aggressiveness "${VAD_WEBRTC_AGGRESSIVENESS}")
  STREAM_ARGS+=(--vad_webrtc_min_voiced_ratio "${VAD_WEBRTC_MIN_VOICED_RATIO}")

  STREAM_ARGS+=(--gain_mode "${GAIN_MODE}")
  STREAM_ARGS+=(--gain_target_delta_db "${GAIN_TARGET_DELTA_DB}")
  STREAM_ARGS+=(--gain_max_boost_db "${GAIN_MAX_BOOST_DB}")
  STREAM_ARGS+=(--gain_smoothing "${GAIN_SMOOTHING}")

  STREAM_ARGS+=(--mask_mode "${MASK_MODE}")
  STREAM_ARGS+=(--mask_db "${MASK_DB}")
  STREAM_ARGS+=(--mask_frame_ms "${MASK_FRAME_MS}")
  STREAM_ARGS+=(--mask_smooth_ms "${MASK_SMOOTH_MS}")
  STREAM_ARGS+=(--peak_limit "${PEAK_LIMIT}")
fi

python -m evaluation.vc_quest.seedvc_playlist_convert \
  --manifest "${MANIFEST}" \
  --out_dir "${RUN_DIR}" \
  --seedvc_dir "${SEEDVC_DIR}" \
  --device "${MODEL_DEVICE}" \
  --seed "${SEED}" \
  "${FP16_ARGS[@]}" \
  --hf_repo "${HF_REPO}" \
  --hf_checkpoint_name "${HF_CHECKPOINT_NAME}" \
  --hf_config_name "${HF_CONFIG_NAME}" \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  --inference_cfg_rate "${INFERENCE_CFG_RATE}" \
  --max_prompt_length_sec "${MAX_PROMPT_SEC}" \
  --length_adjust "${LENGTH_ADJUST}" \
  --max_pairs "${MAX_PAIRS}" \
  "${STREAM_ARGS[@]+"${STREAM_ARGS[@]}"}"

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

echo "[seedvc_run_fleurs_fr_playlist] Wrote ${RUN_DIR}/summary.json"
