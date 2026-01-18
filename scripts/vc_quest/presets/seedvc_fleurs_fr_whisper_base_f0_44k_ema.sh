#!/usr/bin/env bash
set -euo pipefail

# Preset: Seed-VC whisper-base-f0 44k (EMA) on FLEURS fr_fr.
#
# Usage:
#   MAX_PAIRS=50 bash scripts/vc_quest/presets/seedvc_fleurs_fr_whisper_base_f0_44k_ema.sh

cd "$(dirname "$0")/../../.."

export RUN_NAME="${RUN_NAME:-seedvc_whisper_base_f0_44k_ema_w300_h300}"

export HF_CHECKPOINT_NAME="DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ema.pth"
export HF_CONFIG_NAME="config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"

export STREAM_WINDOW_MS="300"
export STREAM_HOP_MS="300"
export DIFFUSION_STEPS="${DIFFUSION_STEPS:-10}"
export INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
export MAX_PROMPT_SEC="${MAX_PROMPT_SEC:-3.0}"

export VAD_MODE="${VAD_MODE:-off}"

export GAIN_MODE="${GAIN_MODE:-off}"
export MASK_MODE="${MASK_MODE:-rms}"
export MASK_DB="${MASK_DB:--50.0}"
export MASK_FRAME_MS="${MASK_FRAME_MS:-10.0}"
export MASK_SMOOTH_MS="${MASK_SMOOTH_MS:-10.0}"

export WER_MODE="${WER_MODE:-audio_ref}"

bash scripts/vc_quest/seedvc_run_fleurs_fr_playlist_gpu.sh

