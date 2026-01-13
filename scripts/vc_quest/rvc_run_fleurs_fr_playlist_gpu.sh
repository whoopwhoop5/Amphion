#!/usr/bin/env bash
set -euo pipefail

# Runs a trained RVC model over the deterministic French playlist (subset to the trained speaker), then scores.
#
# Prereqs:
#   bash scripts/vc_quest/rvc_setup_gpu.sh
#   bash scripts/vc_quest/rvc_train_fleurs_speaker_gpu.sh  (or otherwise produce assets/weights/*.pth)

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="rvc"

cd "$(dirname "$0")/../.."

PLAYLIST_DIR="${PLAYLIST_DIR:-data/vc_quest_playlists/fleurs_fr_fr_dev_v1}"
FULL_MANIFEST="${FULL_MANIFEST:-${PLAYLIST_DIR}/manifest.json}"

SPEAKER_ID="${SPEAKER_ID:-1523}"

RUN_NAME="${RUN_NAME:-rvc_fleurs_fr_s${SPEAKER_ID}_w800_h400_end}"
RUN_DIR="runs/vc_quest/playlists/fleurs_fr_fr/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

SUB_MANIFEST="${SUB_MANIFEST:-${RUN_DIR}/manifest_subset.json}"

RVC_DIR="${RVC_DIR:-${HOME}/deps/rvc_webui}"
MODEL_NAME="${MODEL_NAME:-rvc_fr_fr_s${SPEAKER_ID}_v1_40k_f0_rmvpe.pth}"

INDEX_PATH="${INDEX_PATH:-}"
INDEX_RATE="${INDEX_RATE:-0.0}" # recommended 0 for streaming (upstream reloads index per window)

F0METHOD="${F0METHOD:-rmvpe}"
F0UP_KEY="${F0UP_KEY:-0}"
FILTER_RADIUS="${FILTER_RADIUS:-3}"
RESAMPLE_SR="${RESAMPLE_SR:-0}"
RMS_MIX_RATE="${RMS_MIX_RATE:-1.0}"
PROTECT="${PROTECT:-0.33}"

STREAM_WINDOW_MS="${STREAM_WINDOW_MS:-800}"
STREAM_HOP_MS="${STREAM_HOP_MS:-400}"
STREAM_FADE_MS="${STREAM_FADE_MS:-10}"
EMIT_ALIGN="${EMIT_ALIGN:-end}"
NORMALIZE_ALIGN="${NORMALIZE_ALIGN:-end}"

VAD_MODE="${VAD_MODE:-rms}"
VAD_DB="${VAD_DB:--55}"
VAD_FRAME_MS="${VAD_FRAME_MS:-10}"
VAD_HANGOVER_MS="${VAD_HANGOVER_MS:-200}"
VAD_WEBRTC_AGGRESSIVENESS="${VAD_WEBRTC_AGGRESSIVENESS:-2}"
VAD_WEBRTC_FRAME_MS="${VAD_WEBRTC_FRAME_MS:-30}"
VAD_WEBRTC_MIN_VOICED_RATIO="${VAD_WEBRTC_MIN_VOICED_RATIO:-0.1}"

MASK_MODE="${MASK_MODE:-off}"
MASK_DB="${MASK_DB:--50.0}"
MASK_FRAME_MS="${MASK_FRAME_MS:-10.0}"
MASK_SMOOTH_MS="${MASK_SMOOTH_MS:-10.0}"
PEAK_LIMIT="${PEAK_LIMIT:-0.99}"

MAX_PAIRS="${MAX_PAIRS:-0}"

# 1) Ensure playlist exists (build in current python env; uses HF downloads).
if [[ ! -f "${FULL_MANIFEST}" ]]; then
  scripts/vc_quest/fleurs_fr_build_playlist.sh
fi

# 2) Build a manifest subset for the trained target speaker (absolute wav paths for portability).
SPEAKER_ID="${SPEAKER_ID}" PLAYLIST_DIR="${PLAYLIST_DIR}" FULL_MANIFEST="${FULL_MANIFEST}" SUB_MANIFEST="${SUB_MANIFEST}" python3 - <<'PY'
import json
import os
from pathlib import Path

speaker_id = str(os.environ["SPEAKER_ID"])
playlist_dir = Path(os.environ["PLAYLIST_DIR"]).resolve()
src_manifest = Path(os.environ["FULL_MANIFEST"]).resolve()
dst_manifest = Path(os.environ["SUB_MANIFEST"]).resolve()
dst_manifest.parent.mkdir(parents=True, exist_ok=True)

with src_manifest.open("r", encoding="utf-8") as f:
    m = json.load(f)

targets = [t for t in m.get("targets", []) if str(t.get("speaker_id", "")) == speaker_id]
allowed_tids = {t["id"] for t in targets}
pairs = [p for p in m.get("pairs", []) if p.get("target_id") in allowed_tids]
allowed_sids = {p["source_id"] for p in pairs}
sources = [s for s in m.get("sources", []) if s.get("id") in allowed_sids]

def abs_wav(p):
    if not p:
        return p
    pth = Path(p)
    if pth.is_absolute():
        return str(pth)
    return str((playlist_dir / pth).resolve())

for s in sources:
    s["wav_path"] = abs_wav(s.get("wav_path"))
for t in targets:
    t["wav_path"] = abs_wav(t.get("wav_path"))

out = {
    "meta": {
        **(m.get("meta") or {}),
        "subset": {"target_speaker_id": speaker_id, "num_pairs": len(pairs)},
        "source_manifest": str(src_manifest),
        "playlist_dir": str(playlist_dir),
    },
    "sources": sources,
    "targets": targets,
    "pairs": pairs,
}

dst_manifest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[rvc_run_fleurs_fr_playlist] Wrote subset manifest: {dst_manifest} (pairs={len(pairs)})")
PY

# 3) Convert in the RVC conda env.
source "${CONDA_SH}"
conda activate "${ENV_NAME}"

python -m evaluation.vc_quest.rvc_playlist_convert \
  --manifest "${SUB_MANIFEST}" \
  --out_dir "${RUN_DIR}" \
  --rvc_dir "${RVC_DIR}" \
  --model_name "${MODEL_NAME}" \
  --device cuda:0 \
  --half \
  --f0up_key "${F0UP_KEY}" \
  --f0method "${F0METHOD}" \
  --index_path "${INDEX_PATH}" \
  --index_rate "${INDEX_RATE}" \
  --filter_radius "${FILTER_RADIUS}" \
  --resample_sr "${RESAMPLE_SR}" \
  --rms_mix_rate "${RMS_MIX_RATE}" \
  --protect "${PROTECT}" \
  --stream \
  --window_ms "${STREAM_WINDOW_MS}" \
  --hop_ms "${STREAM_HOP_MS}" \
  --fade_ms "${STREAM_FADE_MS}" \
  --normalize_align "${NORMALIZE_ALIGN}" \
  --emit_align "${EMIT_ALIGN}" \
  --max_pairs "${MAX_PAIRS}" \
  --vad_mode "${VAD_MODE}" \
  --vad_db "${VAD_DB}" \
  --vad_frame_ms "${VAD_FRAME_MS}" \
  --vad_hangover_ms "${VAD_HANGOVER_MS}" \
  --vad_webrtc_aggressiveness "${VAD_WEBRTC_AGGRESSIVENESS}" \
  --vad_webrtc_frame_ms "${VAD_WEBRTC_FRAME_MS}" \
  --vad_webrtc_min_voiced_ratio "${VAD_WEBRTC_MIN_VOICED_RATIO}" \
  --mask_mode "${MASK_MODE}" \
  --mask_db "${MASK_DB}" \
  --mask_frame_ms "${MASK_FRAME_MS}" \
  --mask_smooth_ms "${MASK_SMOOTH_MS}" \
  --peak_limit "${PEAK_LIMIT}"

conda deactivate || true

# 4) Score in Amphion's eval venv.
source .venv/bin/activate

SCORE_ARGS=()
if [[ "${PITCH_METRICS:-0}" == "1" ]]; then
  SCORE_ARGS+=(--pitch_metrics)
fi

WER_MODE="${WER_MODE:-transcript}"

python -m evaluation.vc_quest.score_playlist \
  --manifest "${SUB_MANIFEST}" \
  --run_dir "${RUN_DIR}" \
  --out_json "${RUN_DIR}/summary.json" \
  --whisper_model base \
  --whisper_language fr \
  --wer_mode "${WER_MODE}" \
  "${SCORE_ARGS[@]}"

echo "[rvc_run_fleurs_fr_playlist] Wrote ${RUN_DIR}/summary.json"
