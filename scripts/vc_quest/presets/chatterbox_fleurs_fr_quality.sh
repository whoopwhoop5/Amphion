#!/usr/bin/env bash
set -euo pipefail

# Preset: ChatterboxVC quality/stability reference on FLEURS fr_fr.
# - Goal: best quality + stability (no dropouts/leak) among our tested call-era configs.
# - Note: NOT call-latency suitable under current scoring (latency_p95_ms stays ~800ms).
#
# Usage:
#   MAX_PAIRS=50 bash scripts/vc_quest/presets/chatterbox_fleurs_fr_quality.sh

cd "$(dirname "$0")/../../.."

export RUN_NAME="${RUN_NAME:-chatterbox_preset_quality_w800_h400_s8_mask_gain5}"
export STREAM_WINDOW_MS="800"
export STREAM_HOP_MS="400"
export STREAM_FADE_MS="10"
export EMIT_ALIGN="center"

export CFM_TIMESTEPS="8"
export WATERMARK="${WATERMARK:-1}"

export VAD_MODE="webrtc"
export VAD_DB="-55"
export VAD_FRAME_MS="10"
export VAD_HANGOVER_MS="200"
export VAD_WEBRTC_AGGRESSIVENESS="2"
export VAD_WEBRTC_FRAME_MS="30"
export VAD_WEBRTC_MIN_VOICED_RATIO="0.1"

export GAIN_MODE="match_src_rms"
export GAIN_TARGET_DELTA_DB="5.0"
export GAIN_MAX_BOOST_DB="24.0"
export GAIN_SMOOTHING="0.0"

export MASK_MODE="rms"
export MASK_DB="-50.0"
export MASK_FRAME_MS="10.0"
export MASK_SMOOTH_MS="10.0"

export WER_MODE="${WER_MODE:-audio_ref}"

bash scripts/vc_quest/chatterbox_run_fleurs_fr_playlist_gpu.sh

