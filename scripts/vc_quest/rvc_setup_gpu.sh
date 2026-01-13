#!/usr/bin/env bash
set -euo pipefail

# Installs RVC WebUI (training + inference) into a dedicated conda env on Vast.ai GPU host.
#
# Repo:
# - https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
#
# Notes:
# - This is training-required (per target voice).
# - We intentionally keep this in a separate conda env to avoid polluting Amphion's .venv.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="rvc"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

DEPS_DIR="${HOME}/deps"
RVC_DIR="${DEPS_DIR}/rvc_webui"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[rvc_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[rvc_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

if [[ ! -d "${RVC_DIR}" ]]; then
  echo "[rvc_setup] Cloning RVC WebUI to ${RVC_DIR}"
  git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git "${RVC_DIR}"
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[rvc_setup] Creating conda env ${ENV_NAME} (python=${PYTHON_VERSION})"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# CUDA torch (override with TORCH_INDEX_URL if needed).
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
python -m pip install -U --index-url "${TORCH_INDEX_URL}" \
  torch==2.3.1+cu121 \
  torchaudio==2.3.1+cu121 \
  torchvision==0.18.1+cu121

echo "[rvc_setup] Installing python deps (requirements-py311.txt)"
python -m pip install -U -r "${RVC_DIR}/requirements-py311.txt"

echo "[rvc_setup] Downloading RVC base models (hubert/rmvpe/pretrained)"
python "${RVC_DIR}/tools/download_models.py"

echo "[rvc_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"

