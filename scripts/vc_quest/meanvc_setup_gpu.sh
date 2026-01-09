#!/usr/bin/env bash
set -euo pipefail

# Installs MeanVC into a dedicated conda env on Vast.ai GPU host.
# Uses /opt/miniforge3 (available on Vast images) to avoid Python 3.12 incompatibilities.
#
# MeanVC is a *streaming zero-shot VC* project with a 200ms chunked runtime.
# We use it as a candidate in Amphion's vc_quest harness.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="meanvc"
DEPS_DIR="${HOME}/deps"
MEANVC_DIR="${DEPS_DIR}/MeanVC"

# MeanVC README points to a Google Drive file for the speaker verification checkpoint.
SV_GDRIVE_ID="1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP"
SV_DST_REL="src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[meanvc_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[meanvc_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[meanvc_setup] Creating conda env ${ENV_NAME} (python=3.11)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.11
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch torchaudio torchvision

# Minimal deps for inference (skip pyaudio / wandb).
# Note: MeanVC imports `matplotlib` from src/model/utils.py even for inference.
python -m pip install -U numpy scipy soundfile librosa tqdm webrtcvad safetensors einops x-transformers omegaconf pyyaml transformers accelerate ema_pytorch jiwer huggingface_hub gdown matplotlib

if [[ ! -d "${MEANVC_DIR}" ]]; then
  echo "[meanvc_setup] Cloning MeanVC to ${MEANVC_DIR}"
  git clone --depth 1 https://github.com/ASLP-lab/MeanVC.git "${MEANVC_DIR}"
fi

cd "${MEANVC_DIR}"

echo "[meanvc_setup] Downloading MeanVC inference checkpoints (HF: ASLP-lab/MeanVC)..."
python download_ckpt.py

mkdir -p "$(dirname "${SV_DST_REL}")"
if [[ ! -f "${SV_DST_REL}" ]]; then
  echo "[meanvc_setup] Downloading speaker verification ckpt from Google Drive (id=${SV_GDRIVE_ID})..."
  # gdown handles Drive confirmation pages.
  python -m gdown --id "${SV_GDRIVE_ID}" -O "${SV_DST_REL}"
fi

echo "[meanvc_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"
