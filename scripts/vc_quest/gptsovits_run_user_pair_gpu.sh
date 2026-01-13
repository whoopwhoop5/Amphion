#!/usr/bin/env bash
set -euo pipefail

# GPT-SoVITS user-pair benchmark (ASR→TTS voice-changer experiment).
# Pipeline:
#   src audio -> Whisper transcript -> GPT-SoVITS TTS in target voice (conditioned on ref audio + its transcript)
#
# Important:
# - GPT-SoVITS in this repo is primarily few-shot TTS; the WebUI "voice changer" tab is marked as under construction.
# - This is NOT time-aligned VC, so we score with `score_s2s.py` (speaker similarity + WER + duration ratio).
#
# Usage (Vast):
#   bash scripts/vc_quest/gptsovits_setup_gpu.sh
#   bash scripts/vc_quest/gptsovits_run_user_pair_gpu.sh

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="GPTSoVits"

cd "$(dirname "$0")/../.."

RUN_DIR="runs/vc_quest/gptsovits/user_pair"
mkdir -p "${RUN_DIR}"

GPTSOVITS_DIR="${HOME}/deps/GPT-SoVITS"
if [ ! -d "${GPTSOVITS_DIR}" ]; then
  echo "[gptsovits_run] Missing GPT-SoVITS repo at ${GPTSOVITS_DIR}. Run: bash scripts/vc_quest/gptsovits_setup_gpu.sh"
  exit 1
fi

REF_V5="assets/vevo_live/user/ref_v5_24k_10s.wav"
SRC_V5="assets/vevo_live/user/src_v5_24k.wav"
REF_FR="assets/vevo_live/user/ref_fr_female_24k_10s.wav"
SRC_FR="assets/vevo_live/user/src_fr_female_24k.wav"

WHISPER_MODEL="${WHISPER_MODEL:-base}"
WHISPER_LANGUAGE="${WHISPER_LANGUAGE:-fr}"

TEXT_LANG="${TEXT_LANG:-en}"
PROMPT_LANG="${PROMPT_LANG:-en}"

echo "[gptsovits_run] Transcribing src/ref using Whisper (${WHISPER_MODEL}, lang=${WHISPER_LANGUAGE})"
source .venv/bin/activate
python - <<PY
import json
from pathlib import Path

from evaluation.vevo_live.common import load_whisper

run_dir = Path("${RUN_DIR}")
run_dir.mkdir(parents=True, exist_ok=True)

whisper_model = load_whisper("${WHISPER_MODEL}")
lang = "${WHISPER_LANGUAGE}".strip() or None

def transcribe(in_wav: str, out_txt: str):
    out_p = Path(out_txt)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    asr = whisper_model.transcribe(in_wav, verbose=False, language=lang) if lang else whisper_model.transcribe(in_wav, verbose=False)
    out_p.write_text(str(asr.get("text") or "").strip(), encoding="utf-8")

for stem, wav in [
    ("ref_fr", "${REF_FR}"),
    ("src_v5", "${SRC_V5}"),
    ("ref_v5", "${REF_V5}"),
    ("src_fr", "${SRC_FR}"),
]:
    transcribe(wav, str(run_dir / f"{stem}.txt"))

print(json.dumps({"wrote": [str(run_dir / f"{s}.txt") for s in ["ref_fr","src_v5","ref_v5","src_fr"]]}, indent=2))
PY
deactivate || true

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

echo "[gptsovits_run] Running GPT-SoVITS TTS (text_lang=${TEXT_LANG}, prompt_lang=${PROMPT_LANG})"

python -m evaluation.vc_quest.gptsovits_tts_infer \
  --gptsovits_dir "${GPTSOVITS_DIR}" \
  --ref_audio_path "${REF_FR}" \
  --prompt_text_file "${RUN_DIR}/ref_fr.txt" \
  --prompt_lang "${PROMPT_LANG}" \
  --text_file "${RUN_DIR}/src_v5.txt" \
  --text_lang "${TEXT_LANG}" \
  --out_wav "${RUN_DIR}/v5_to_fr_s2s.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_s2s.meta.json"

python -m evaluation.vc_quest.gptsovits_tts_infer \
  --gptsovits_dir "${GPTSOVITS_DIR}" \
  --ref_audio_path "${REF_V5}" \
  --prompt_text_file "${RUN_DIR}/ref_v5.txt" \
  --prompt_lang "${PROMPT_LANG}" \
  --text_file "${RUN_DIR}/src_fr.txt" \
  --text_lang "${TEXT_LANG}" \
  --out_wav "${RUN_DIR}/fr_to_v5_s2s.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_s2s.meta.json"

conda deactivate || true

echo "[gptsovits_run] Scoring outputs (speaker sim + WER + duration ratio)"
source .venv/bin/activate

python -m evaluation.vc_quest.score_s2s \
  --ref_wav "${REF_FR}" \
  --src_wav "${SRC_V5}" \
  --deg_wav "${RUN_DIR}/v5_to_fr_s2s.wav" \
  --meta_json "${RUN_DIR}/v5_to_fr_s2s.meta.json" \
  --src_text_file "${RUN_DIR}/src_v5.txt" \
  --out_json "${RUN_DIR}/v5_to_fr_s2s.report.json" \
  --whisper_model "${WHISPER_MODEL}" \
  --whisper_language "${WHISPER_LANGUAGE}"

python -m evaluation.vc_quest.score_s2s \
  --ref_wav "${REF_V5}" \
  --src_wav "${SRC_FR}" \
  --deg_wav "${RUN_DIR}/fr_to_v5_s2s.wav" \
  --meta_json "${RUN_DIR}/fr_to_v5_s2s.meta.json" \
  --src_text_file "${RUN_DIR}/src_fr.txt" \
  --out_json "${RUN_DIR}/fr_to_v5_s2s.report.json" \
  --whisper_model "${WHISPER_MODEL}" \
  --whisper_language "${WHISPER_LANGUAGE}"

echo "[gptsovits_run] Wrote artifacts to ${RUN_DIR}"

