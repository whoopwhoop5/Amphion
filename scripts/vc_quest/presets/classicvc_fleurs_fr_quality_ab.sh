#!/usr/bin/env bash
set -euo pipefail

# Preset: ClassicVC/MMCXLI quality A/B on FLEURS fr_fr.
# - Runs the existing windowed streaming sim vs MMCXLI's native realtime inference path.
#
# Usage:
#   MAX_PAIRS=50 bash scripts/vc_quest/presets/classicvc_fleurs_fr_quality_ab.sh

cd "$(dirname "$0")/../../.."

export STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-2000}"
export STREAM_HOP_MS="${STREAM_HOP_MS:-400}"
export STREAM_FADE_MS="${STREAM_FADE_MS:-10}"
export EMIT_ALIGN="${EMIT_ALIGN:-end}"

export VAD_MODE="${VAD_MODE:-rms}"
export VAD_DB="${VAD_DB:--55}"
export VAD_FRAME_MS="${VAD_FRAME_MS:-10}"
export VAD_HANGOVER_MS="${VAD_HANGOVER_MS:-200}"

export REF_VAD_MODE="${REF_VAD_MODE:-rms}"

export GAIN_MODE="${GAIN_MODE:-off}"
export MASK_MODE="${MASK_MODE:-off}"

export WER_MODE="${WER_MODE:-audio_ref}"

export STREAM_BACKEND="windowed"
export RUN_NAME="${RUN_NAME_WINDOWED:-classicvc_preset_quality_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}_${EMIT_ALIGN}_windowed}"
bash scripts/vc_quest/classicvc_run_fleurs_fr_playlist_gpu.sh

export STREAM_BACKEND="mmcxli_infer"
export RUN_NAME="${RUN_NAME_INFER:-classicvc_preset_quality_w${STREAM_WINDOW_MS}_h${STREAM_HOP_MS}_${EMIT_ALIGN}_infer}"
bash scripts/vc_quest/classicvc_run_fleurs_fr_playlist_gpu.sh
