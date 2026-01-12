#!/usr/bin/env bash
set -euo pipefail

# Preset: FreeVC quality-first on FLEURS fr_fr.
# - Goal: better content preservation than the call preset.
# - Tradeoff: higher algorithmic delay than emit_align=end.
#
# Usage:
#   MAX_PAIRS=50 bash scripts/vc_quest/presets/freevc_fleurs_fr_quality.sh

cd "$(dirname "$0")/../../.."

export RUN_NAME="${RUN_NAME:-freevc_preset_quality_w800_h400_center}"
export STREAM_WINDOW_MS="800"
export STREAM_HOP_MS="400"
export STREAM_FADE_MS="10"
export EMIT_ALIGN="center"

export VAD_MODE="webrtc"
export VAD_DB="-55"
export VAD_FRAME_MS="10"
export VAD_HANGOVER_MS="200"
export VAD_WEBRTC_AGGRESSIVENESS="2"
export VAD_WEBRTC_FRAME_MS="30"
export VAD_WEBRTC_MIN_VOICED_RATIO="0.1"

# Keep these off by default: not validated on full 300-pair runs yet.
export GAIN_MODE="off"
export MASK_MODE="off"

export WER_MODE="${WER_MODE:-audio_ref}"

bash scripts/vc_quest/freevc_run_fleurs_fr_playlist_gpu.sh

