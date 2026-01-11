#!/usr/bin/env bash
set -euo pipefail

# Deterministic grid search for FreeVC streaming parameters on the user pair.
# Generates offline outputs once, then evaluates multiple (window_ms, hop_ms) settings for streaming.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="freevc"

cd "$(dirname "$0")/../.."

RUN_DIR="${RUN_DIR:-runs/vc_quest/freevc/user_pair_search_webrtc_center}"
mkdir -p "${RUN_DIR}"

FREEVC_DIR="${HOME}/deps/FreeVC"

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

VARIANT="${VARIANT:-freevc-24}"
SEED="${SEED:-0}"

# Streaming grid (override via env).
WINDOWS_MS="${WINDOWS_MS:-600 800 1000 1200}"
HOPS_MS="${HOPS_MS:-100 150 200 300 400}"
FADE_MS="${FADE_MS:-10}"

VAD_DB="${VAD_DB:--55}"
VAD_FRAME_MS="${VAD_FRAME_MS:-10}"
VAD_MODE="${VAD_MODE:-webrtc}"
VAD_HANGOVER_MS="${VAD_HANGOVER_MS:-200}"
VAD_WEBRTC_AGGRESSIVENESS="${VAD_WEBRTC_AGGRESSIVENESS:-2}"
VAD_WEBRTC_FRAME_MS="${VAD_WEBRTC_FRAME_MS:-30}"
VAD_WEBRTC_MIN_VOICED_RATIO="${VAD_WEBRTC_MIN_VOICED_RATIO:-0.1}"
EMIT_ALIGN="${EMIT_ALIGN:-center}"
PEAK_LIMIT="${PEAK_LIMIT:-0.99}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

echo "[freevc_search] freevc_dir=${FREEVC_DIR} variant=${VARIANT}"
echo "[freevc_search] grid windows_ms=[${WINDOWS_MS}] hops_ms=[${HOPS_MS}]"
echo "[freevc_search] vad_mode=${VAD_MODE} emit_align=${EMIT_ALIGN}"

# Offline once (used as a quality reference for listening).
python -m evaluation.vc_quest.freevc_convert \
  --freevc_dir "${FREEVC_DIR}" \
  --variant "${VARIANT}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref "${REF_FR}" \
  --src "${SRC_V5}" \
  --out "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json"

python -m evaluation.vc_quest.freevc_convert \
  --freevc_dir "${FREEVC_DIR}" \
  --variant "${VARIANT}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --ref "${REF_V5}" \
  --src "${SRC_FR}" \
  --out "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json"

# Streaming grid.
for window_ms in ${WINDOWS_MS}; do
  for hop_ms in ${HOPS_MS}; do
    if [[ "${hop_ms}" -gt "${window_ms}" ]]; then
      continue
    fi
    tag="w${window_ms}_h${hop_ms}"
    echo "[freevc_search] stream ${tag}"

    python -m evaluation.vc_quest.freevc_convert \
      --freevc_dir "${FREEVC_DIR}" \
      --variant "${VARIANT}" \
      --device cuda:0 \
      --seed "${SEED}" \
      --ref "${REF_FR}" \
      --src "${SRC_V5}" \
      --out "${RUN_DIR}/v5_to_fr_stream_${tag}.wav" \
      --meta_json "${RUN_DIR}/v5_to_fr_stream_${tag}.meta.json" \
      --stream \
      --window_ms "${window_ms}" \
      --hop_ms "${hop_ms}" \
      --fade_ms "${FADE_MS}" \
      --emit_align "${EMIT_ALIGN}" \
      --vad_mode "${VAD_MODE}" \
      --vad_db "${VAD_DB}" \
      --vad_frame_ms "${VAD_FRAME_MS}" \
      --vad_hangover_ms "${VAD_HANGOVER_MS}" \
      --vad_webrtc_aggressiveness "${VAD_WEBRTC_AGGRESSIVENESS}" \
      --vad_webrtc_frame_ms "${VAD_WEBRTC_FRAME_MS}" \
      --vad_webrtc_min_voiced_ratio "${VAD_WEBRTC_MIN_VOICED_RATIO}" \
      --peak_limit "${PEAK_LIMIT}"

    python -m evaluation.vc_quest.freevc_convert \
      --freevc_dir "${FREEVC_DIR}" \
      --variant "${VARIANT}" \
      --device cuda:0 \
      --seed "${SEED}" \
      --ref "${REF_V5}" \
      --src "${SRC_FR}" \
      --out "${RUN_DIR}/fr_to_v5_stream_${tag}.wav" \
      --meta_json "${RUN_DIR}/fr_to_v5_stream_${tag}.meta.json" \
      --stream \
      --window_ms "${window_ms}" \
      --hop_ms "${hop_ms}" \
      --fade_ms "${FADE_MS}" \
      --emit_align "${EMIT_ALIGN}" \
      --vad_mode "${VAD_MODE}" \
      --vad_db "${VAD_DB}" \
      --vad_frame_ms "${VAD_FRAME_MS}" \
      --vad_hangover_ms "${VAD_HANGOVER_MS}" \
      --vad_webrtc_aggressiveness "${VAD_WEBRTC_AGGRESSIVENESS}" \
      --vad_webrtc_frame_ms "${VAD_WEBRTC_FRAME_MS}" \
      --vad_webrtc_min_voiced_ratio "${VAD_WEBRTC_MIN_VOICED_RATIO}" \
      --peak_limit "${PEAK_LIMIT}"
  done
done

conda deactivate || true
source .venv/bin/activate

echo "[freevc_search] Scoring outputs..."

SCORE_ARGS=()
if [[ "${PITCH_METRICS:-0}" == "1" ]]; then
  SCORE_ARGS+=(--pitch_metrics)
fi

python -m evaluation.vc_quest.score_outputs \
  --ref_wav "${REF_FR}" \
  --src_wav "${SRC_V5}" \
  --deg_wav "${RUN_DIR}/v5_to_fr_offline.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_offline.meta.json" \
  --out_json "${RUN_DIR}/v5_to_fr_offline.report.json" \
  "${SCORE_ARGS[@]}"

python -m evaluation.vc_quest.score_outputs \
  --ref_wav "${REF_V5}" \
  --src_wav "${SRC_FR}" \
  --deg_wav "${RUN_DIR}/fr_to_v5_offline.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_offline.meta.json" \
  --out_json "${RUN_DIR}/fr_to_v5_offline.report.json" \
  "${SCORE_ARGS[@]}"

for window_ms in ${WINDOWS_MS}; do
  for hop_ms in ${HOPS_MS}; do
    if [[ "${hop_ms}" -gt "${window_ms}" ]]; then
      continue
    fi
    tag="w${window_ms}_h${hop_ms}"
    python -m evaluation.vc_quest.score_outputs \
      --ref_wav "${REF_FR}" \
      --src_wav "${SRC_V5}" \
      --deg_wav "${RUN_DIR}/v5_to_fr_stream_${tag}.wav" \
      --meta_json "${RUN_DIR}/v5_to_fr_stream_${tag}.meta.json" \
      --out_json "${RUN_DIR}/v5_to_fr_stream_${tag}.report.json" \
      "${SCORE_ARGS[@]}"

    python -m evaluation.vc_quest.score_outputs \
      --ref_wav "${REF_V5}" \
      --src_wav "${SRC_FR}" \
      --deg_wav "${RUN_DIR}/fr_to_v5_stream_${tag}.wav" \
      --meta_json "${RUN_DIR}/fr_to_v5_stream_${tag}.meta.json" \
      --out_json "${RUN_DIR}/fr_to_v5_stream_${tag}.report.json" \
      "${SCORE_ARGS[@]}"
  done
done

python -m evaluation.vc_quest.select_best --run_dir "${RUN_DIR}"

echo "[freevc_search] Done. Artifacts in ${RUN_DIR}"
