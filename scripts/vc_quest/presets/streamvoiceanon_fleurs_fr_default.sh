#!/usr/bin/env bash
set -euo pipefail

# Preset: StreamVoiceAnon default-ish streaming config on FLEURS fr_fr.
#
# Usage:
#   MAX_PAIRS=50 bash scripts/vc_quest/presets/streamvoiceanon_fleurs_fr_default.sh

cd "$(dirname "$0")/../../.."

export RUN_NAME="${RUN_NAME:-streamvoiceanon_preset_d2_c1_e128_d64}"
export ENCODE_WINDOW_FRAMES="${ENCODE_WINDOW_FRAMES:-128}"
export DECODE_WINDOW_FRAMES="${DECODE_WINDOW_FRAMES:-64}"
export MAX_PROMPT_FRAMES="${MAX_PROMPT_FRAMES:-256}"
export MAX_SEQ_FRAMES="${MAX_SEQ_FRAMES:-768}"
export BUFFER_FRAMES="${BUFFER_FRAMES:-32}"
export DECODE_CHUNK_FRAMES="${DECODE_CHUNK_FRAMES:-1}"
export DELAY_FRAMES="${DELAY_FRAMES:-2}"

# StreamVoiceAnon README recommends compile for realtime; keep it optional.
export COMPILE_AR="${COMPILE_AR:-1}"
export COMPILE_ENCODER="${COMPILE_ENCODER:-1}"
export COMPILE_DECODER="${COMPILE_DECODER:-1}"

export FADE_MS="${FADE_MS:-10}"
export GAIN_MODE="${GAIN_MODE:-off}"
export MASK_MODE="${MASK_MODE:-off}"

export WER_MODE="${WER_MODE:-audio_ref}"

bash scripts/vc_quest/streamvoiceanon_run_fleurs_fr_playlist_gpu.sh

