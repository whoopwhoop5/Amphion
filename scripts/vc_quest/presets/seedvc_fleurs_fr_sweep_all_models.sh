#!/usr/bin/env bash
set -euo pipefail

# Sweep: run all Seed-VC HF checkpoint variants we can load in our wrapper on FLEURS fr_fr.
#
# Usage:
#   MAX_PAIRS=50 bash scripts/vc_quest/presets/seedvc_fleurs_fr_sweep_all_models.sh
#   MAX_PAIRS=0  bash scripts/vc_quest/presets/seedvc_fleurs_fr_sweep_all_models.sh

cd "$(dirname "$0")/../../.."

export STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-300}"
export STREAM_HOP_MS="${STREAM_HOP_MS:-300}"
export DIFFUSION_STEPS="${DIFFUSION_STEPS:-10}"
export INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
export MAX_PROMPT_SEC="${MAX_PROMPT_SEC:-3.0}"

# Defaults: avoid VAD-induced dropouts and suppress silence noise.
export VAD_MODE="${VAD_MODE:-off}"
export MASK_MODE="${MASK_MODE:-rms}"
export MASK_DB="${MASK_DB:--50.0}"
export MASK_FRAME_MS="${MASK_FRAME_MS:-10.0}"
export MASK_SMOOTH_MS="${MASK_SMOOTH_MS:-10.0}"

export WER_MODE="${WER_MODE:-audio_ref}"

declare -a tags=(
  "xlsr_tiny"
  "whisper_small_wavenet"
  "whisper_base_f0_44k_ema"
  "whisper_base_f0_44k_ft_ema"
  "whisper_base_f0_44k_ft_ema_v2"
)

declare -a checkpoints=(
  "DiT_uvit_tat_xlsr_ema.pth"
  "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
  "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ema.pth"
  "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth"
  "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
)

declare -a configs=(
  "config_dit_mel_seed_uvit_xlsr_tiny.yml"
  "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
  "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
  "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
  "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
)

if [[ "${#tags[@]}" -ne "${#checkpoints[@]}" || "${#tags[@]}" -ne "${#configs[@]}" ]]; then
  echo "[seedvc_sweep] internal error: array length mismatch" >&2
  exit 1
fi

for i in "${!tags[@]}"; do
  export HF_CHECKPOINT_NAME="${checkpoints[$i]}"
  export HF_CONFIG_NAME="${configs[$i]}"

  export RUN_NAME="seedvc_${tags[$i]}_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}_s${DIFFUSION_STEPS}_cfg${INFERENCE_CFG_RATE}_${VAD_MODE}_${MASK_MODE}"
  echo "[seedvc_sweep] RUN_NAME=${RUN_NAME} ckpt=${HF_CHECKPOINT_NAME} cfg=${HF_CONFIG_NAME}"

  bash scripts/vc_quest/seedvc_run_fleurs_fr_playlist_gpu.sh
done

