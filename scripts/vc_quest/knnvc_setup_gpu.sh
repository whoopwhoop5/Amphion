#!/usr/bin/env bash
set -euo pipefail

# Installs kNN-VC into a dedicated conda env on Vast.ai GPU host.
# Uses /opt/miniforge3 (available on Vast images) to avoid Python 3.12 incompatibilities.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="knnvc"
DEPS_DIR="${HOME}/deps"
KNNVC_DIR="${DEPS_DIR}/knn-vc"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[knnvc_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[knnvc_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[knnvc_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch torchaudio torchvision

# Minimal deps for our wrapper + resampling.
python -m pip install -U numpy scipy soundfile librosa webrtcvad

if [[ ! -d "${KNNVC_DIR}" ]]; then
  echo "[knnvc_setup] Cloning knn-vc to ${KNNVC_DIR}"
  git clone --depth 1 https://github.com/bshall/knn-vc.git "${KNNVC_DIR}"
fi

export KNNVC_DIR

echo "[knnvc_setup] Pre-downloading knn-vc weights (torch hub cache)..."
python - <<'PY'
import os
import torch

repo_dir = os.environ["KNNVC_DIR"]
device = "cuda:0" if torch.cuda.is_available() else "cpu"

_ = torch.hub.load(
    repo_dir,
    "knn_vc",
    source="local",
    pretrained=True,
    prematched=True,
    device=device,
)
print("[knnvc_setup] OK: knn-vc weights cached")
PY

echo "[knnvc_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"

