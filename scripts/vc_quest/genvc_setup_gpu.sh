#!/usr/bin/env bash
set -euo pipefail

# Installs GenVC into a dedicated conda env on Vast.ai GPU host.
# Repo: https://github.com/caizexin/GenVC
# Weights: https://huggingface.co/ZexinCai/GenVC (requires HF login for best reliability)

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="genvc"
DEPS_DIR="${HOME}/deps"
GENVC_DIR="${DEPS_DIR}/GenVC"
HF_REPO_ID="ZexinCai/GenVC"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[genvc_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[genvc_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

export HF_HOME="${HOME}/.hf_home"
mkdir -p "${HF_HOME}"

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[genvc_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# Torch pinned to the Vast RTX 4090 CUDA stack (cu121).
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchaudio==2.5.1

# GenVC author-recommended transformers version for streaming stability.
python -m pip install -U transformers==4.33.0

# fairseq (and its hydra/omegaconf deps) requires pip<24.1 due to legacy metadata on omegaconf<2.1.
python -m pip install -U "pip<24.1"

# fairseq is required for ContentVec feature extraction (python<3.11).
python -m pip install -U "fairseq==0.12.2"

if [[ ! -d "${GENVC_DIR}" ]]; then
  echo "[genvc_setup] Cloning GenVC to ${GENVC_DIR}"
  git clone --depth 1 https://github.com/caizexin/GenVC.git "${GENVC_DIR}"
fi

python -m pip install -U -r "${GENVC_DIR}/requirements.txt"

echo "[genvc_setup] Downloading pretrained files from HF (${HF_REPO_ID}) into ${GENVC_DIR} ..."
python - <<PY
import os
from huggingface_hub import snapshot_download

repo_id = os.environ.get("GENVC_HF_REPO_ID", "${HF_REPO_ID}")
local_dir = os.environ.get("GENVC_DIR", "${GENVC_DIR}")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    allow_patterns=["pre_trained/*"],
)
print("OK downloaded: pre_trained/*")
PY

echo "[genvc_setup] pre_trained contents:"
ls -la "${GENVC_DIR}/pre_trained" | head -n 50

echo "[genvc_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"
