#!/usr/bin/env bash
set -euo pipefail

# Installs EZ-VC into a dedicated conda env on Vast.ai GPU host.
# Notes:
# - HF weights for the EZ-VC model are gated (SPRINGLab/EZ-VC). You must run:
#     huggingface-cli login
#   once on the GPU host before downloads will work.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="ezvc"
DEPS_DIR="${HOME}/deps"
EZVC_DIR="${DEPS_DIR}/EZ-VC"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[ezvc_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ezvc_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[ezvc_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch torchaudio torchvision

if [[ ! -d "${EZVC_DIR}" ]]; then
  echo "[ezvc_setup] Cloning EZ-VC to ${EZVC_DIR}"
  git clone https://github.com/EZ-VC/EZ-VC.git "${EZVC_DIR}"
fi

cd "${EZVC_DIR}"
git submodule update --init --recursive

echo "[ezvc_setup] Installing EZ-VC (may take a while)..."

# Prefer upstream deps, but retry without deps if optional heavy deps (e.g. bitsandbytes) fail.
python -m pip install -e . || {
  echo "[ezvc_setup] WARNING: pip install -e . failed; retrying with --no-deps + minimal deps" >&2
  python -m pip install -e . --no-deps
  python -m pip install -U \
    "numpy==1.26.4" \
    "pydantic<=2.10.6" \
    accelerate cached_path click datasets ema_pytorch hydra-core jieba librosa matplotlib pydub pypinyin \
    safetensors soundfile tomli torchdiffeq tqdm transformers transformers_stream_generator unidecode vocos wandb \
    "x_transformers>=1.31.14"
}

echo "[ezvc_setup] Installing espnet fork (required for XEUS units)..."
python -m pip install -U 'espnet @ git+https://github.com/wanchichen/espnet.git@ssl'

echo "[ezvc_setup] Pre-downloading public dependencies (XEUS + BigVGAN vocoder weights)..."
python - <<'PY'
from huggingface_hub import hf_hub_download

_ = hf_hub_download("espnet/xeus", filename="model/xeus_checkpoint_old.pth")
_ = hf_hub_download("SPRINGLab/bigvgan_16khz", filename="pytorch_model.bin")
print("[ezvc_setup] OK")
PY

echo "[ezvc_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"
