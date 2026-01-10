#!/usr/bin/env bash
set -euo pipefail

# Installs TinyVC into a dedicated conda env on Vast.ai GPU host.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="tinyvc"
DEPS_DIR="${HOME}/deps"
TINYVC_DIR="${DEPS_DIR}/tinyvc"
HF_REPO_ID="uthree/tinyvc"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[tinyvc_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[tinyvc_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

export HF_HOME="${HOME}/.hf_home"
mkdir -p "${HF_HOME}"

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[tinyvc_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch torchaudio torchvision

# Minimal deps for inference.
python -m pip install -U numpy scipy soundfile tqdm webrtcvad huggingface_hub pyworld torchfcpe

if [[ ! -d "${TINYVC_DIR}" ]]; then
  echo "[tinyvc_setup] Cloning tinyvc to ${TINYVC_DIR}"
  git clone --depth 1 https://github.com/uthree/tinyvc.git "${TINYVC_DIR}"
else
  echo "[tinyvc_setup] Updating tinyvc under ${TINYVC_DIR}"
  git -C "${TINYVC_DIR}" fetch --depth 1 origin
  git -C "${TINYVC_DIR}" checkout -q origin/main
fi

export TINYVC_DIR
export HF_REPO_ID

echo "[tinyvc_setup] Downloading pretrained weights (HF: ${HF_REPO_ID})..."
python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id = os.environ.get("HF_REPO_ID", "uthree/tinyvc")
root = Path(os.environ["TINYVC_DIR"]).resolve()

files = [
    "models/encoder.pt",
    "models/decoder.pt",
]

for rel in files:
    src = hf_hub_download(repo_id, filename=rel)
    dst = root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"[tinyvc_setup] OK: {rel}")
PY

echo "[tinyvc_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"

