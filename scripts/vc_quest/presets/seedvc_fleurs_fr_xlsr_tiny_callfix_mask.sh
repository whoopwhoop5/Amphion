#!/usr/bin/env bash
set -euo pipefail

# Preset: Seed-VC XLSR-tiny with VAD off + RMS mask (attempt to reduce dropouts/leakage).
#
# Usage:
#   MAX_PAIRS=50 bash scripts/vc_quest/presets/seedvc_fleurs_fr_xlsr_tiny_callfix_mask.sh

cd "$(dirname "$0")/../../.."

export RUN_NAME="${RUN_NAME:-seedvc_xlsr_tiny_w300_h300_vadoff_mask}"

export HF_CHECKPOINT_NAME="DiT_uvit_tat_xlsr_ema.pth"
export HF_CONFIG_NAME="config_dit_mel_seed_uvit_xlsr_tiny.yml"

export STREAM_WINDOW_MS="300"
export STREAM_HOP_MS="300"
export DIFFUSION_STEPS="${DIFFUSION_STEPS:-10}"
export INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
export MAX_PROMPT_SEC="${MAX_PROMPT_SEC:-3.0}"

export VAD_MODE="off"

export GAIN_MODE="${GAIN_MODE:-off}"
export MASK_MODE="rms"
export MASK_DB="${MASK_DB:--50.0}"
export MASK_FRAME_MS="${MASK_FRAME_MS:-10.0}"
export MASK_SMOOTH_MS="${MASK_SMOOTH_MS:-10.0}"

export WER_MODE="${WER_MODE:-audio_ref}"

bash scripts/vc_quest/seedvc_run_fleurs_fr_playlist_gpu.sh

