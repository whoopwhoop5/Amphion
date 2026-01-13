#!/usr/bin/env bash
set -euo pipefail

# Setup GPT-SoVITS on a GPU host (Vast).
# Notes:
# - GPT-SoVITS is primarily few-shot TTS; its "voice changer" tab is currently marked as under construction.
# - We still set it up to benchmark an ASR→TTS voice-changer pipeline.
#
# Usage (Vast):
#   DEVICE=CU128 SOURCE=HF bash scripts/vc_quest/gptsovits_setup_gpu.sh

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="GPTSoVits"

cd "$(dirname "$0")/../.."

DEPS_DIR="${HOME}/deps"
GPTSOVITS_DIR="${DEPS_DIR}/GPT-SoVITS"

DEVICE="${DEVICE:-CU128}"
SOURCE="${SOURCE:-HF}"

mkdir -p "${DEPS_DIR}"
if [ ! -d "${GPTSOVITS_DIR}" ]; then
  git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git "${GPTSOVITS_DIR}"
fi

source "${CONDA_SH}"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"
cd "${GPTSOVITS_DIR}"

# install.sh uses `tput` for UX; make it work under nohup/non-interactive shells.
export TERM="${TERM:-xterm}"

# Some minimal environments are missing terminfo entries; make install.sh robust to tput failures.
if grep -q "tput cuu1 && tput el" install.sh 2>/dev/null; then
  sed -i 's/tput cuu1 && tput el/(tput cuu1 && tput el) || true/' install.sh
fi

# Install deps + download pretrained models.
bash install.sh --device "${DEVICE}" --source "${SOURCE}"

# torchaudio 2.9 uses TorchCodec for loading audio; install.sh doesn't always install it.
python -m pip install --upgrade --quiet torchcodec

if ! command -v ffmpeg >/dev/null 2>&1; then
  conda install -y ffmpeg
fi

echo "[gptsovits_setup] Done. Repo=${GPTSOVITS_DIR} env=${ENV_NAME} device=${DEVICE} source=${SOURCE}"
