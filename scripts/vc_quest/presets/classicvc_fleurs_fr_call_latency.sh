#!/usr/bin/env bash
set -euo pipefail

# Preset: ClassicVC/MMCXLI call-latency-first on FLEURS fr_fr.
# - Goal: strong call UX (low algorithmic delay + very fast inference) while staying stable.
#
# Usage:
#   MAX_PAIRS=50 bash scripts/vc_quest/presets/classicvc_fleurs_fr_call_latency.sh

cd "$(dirname "$0")/../../.."

export RUN_NAME="${RUN_NAME:-classicvc_preset_call_w800_h400_end}"
export STREAM_WINDOW_MS="800"
export STREAM_HOP_MS="400"
export STREAM_FADE_MS="10"
export EMIT_ALIGN="end"

# Match the validated Vast run defaults.
export VAD_MODE="rms"
export VAD_DB="-55"
export VAD_FRAME_MS="10"
export VAD_HANGOVER_MS="200"

export REF_VAD_MODE="${REF_VAD_MODE:-rms}"

export GAIN_MODE="off"
export MASK_MODE="off"

export WER_MODE="${WER_MODE:-audio_ref}"

bash scripts/vc_quest/classicvc_run_fleurs_fr_playlist_gpu.sh
