#!/usr/bin/env bash
set -euo pipefail

# Preset: ClassicVC/MMCXLI quality-first on FLEURS fr_fr (still call-usable).
# - Goal: improve intelligibility/quality while keeping latency_p95_ms < 500ms.
#
# Usage:
#   MAX_PAIRS=50 bash scripts/vc_quest/presets/classicvc_fleurs_fr_quality.sh

cd "$(dirname "$0")/../../.."

export RUN_NAME="${RUN_NAME:-classicvc_preset_quality_w1600_h400_end}"
export STREAM_WINDOW_MS="1600"
export STREAM_HOP_MS="400"
export STREAM_FADE_MS="10"
export EMIT_ALIGN="end"

# Match the validated Vast run defaults.
export VAD_MODE="rms"
export VAD_DB="-55"
export VAD_FRAME_MS="10"
export VAD_HANGOVER_MS="200"

export GAIN_MODE="off"
export MASK_MODE="off"

export WER_MODE="${WER_MODE:-audio_ref}"

bash scripts/vc_quest/classicvc_run_fleurs_fr_playlist_gpu.sh
