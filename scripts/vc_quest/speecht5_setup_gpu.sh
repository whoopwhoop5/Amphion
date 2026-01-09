#!/usr/bin/env bash
set -euo pipefail

# Installs SpeechT5 VC into a dedicated conda env on Vast.ai GPU host.
# Uses /opt/miniforge3 (available on Vast images) to avoid Python 3.12 incompatibilities.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="speecht5"

VC_MODEL="${VC_MODEL:-microsoft/speecht5_vc}"
VOCODER_MODEL="${VOCODER_MODEL:-microsoft/speecht5_hifigan}"
SPEAKER_MODEL="${SPEAKER_MODEL:-microsoft/wavlm-base-plus-sv}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[speecht5_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[speecht5_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[speecht5_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch torchaudio torchvision

# Minimal deps for our wrapper.
python -m pip install -U numpy scipy soundfile librosa webrtcvad "transformers==4.41.2"

export VC_MODEL VOCODER_MODEL SPEAKER_MODEL

echo "[speecht5_setup] Pre-downloading models into HF cache..."
python - <<'PY'
import os

from transformers import (
    AutoFeatureExtractor,
    SpeechT5ForSpeechToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
    WavLMForXVector,
)

vc_model = os.environ.get("VC_MODEL", "microsoft/speecht5_vc")
vocoder_model = os.environ.get("VOCODER_MODEL", "microsoft/speecht5_hifigan")
speaker_model = os.environ.get("SPEAKER_MODEL", "microsoft/wavlm-base-plus-sv")

_ = SpeechT5Processor.from_pretrained(vc_model)
_ = SpeechT5ForSpeechToSpeech.from_pretrained(vc_model)
_ = SpeechT5HifiGan.from_pretrained(vocoder_model)
_ = AutoFeatureExtractor.from_pretrained(speaker_model)
_ = WavLMForXVector.from_pretrained(speaker_model)

print("[speecht5_setup] OK: models cached")
PY

echo "[speecht5_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"

