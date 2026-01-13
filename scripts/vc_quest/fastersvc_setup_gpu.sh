#!/usr/bin/env bash
set -euo pipefail

# Installs FasterSVC into a dedicated conda env on Vast.ai GPU host.
# Uses /opt/miniforge3 (available on Vast images) to avoid Python 3.12 incompatibilities.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="fastersvc"
DEPS_DIR="${HOME}/deps"
FASTER_SVC_DIR="${DEPS_DIR}/fastersvc"

HF_REPO="uthree/fastersvc-jvs-corpus-pretrained"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[fastersvc_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[fastersvc_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[fastersvc_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch torchaudio torchvision

# Minimal deps for inference + our streaming eval wrappers.
python -m pip install -U numpy scipy soundfile librosa tqdm webrtcvad pyworld huggingface_hub

if [[ ! -d "${FASTER_SVC_DIR}" ]]; then
  echo "[fastersvc_setup] Cloning FasterSVC to ${FASTER_SVC_DIR}"
  git clone --depth 1 https://github.com/uthree/fastersvc.git "${FASTER_SVC_DIR}"
fi

cd "${FASTER_SVC_DIR}"
mkdir -p models

export HF_REPO
export FASTER_SVC_DIR

echo "[fastersvc_setup] Downloading pretrained weights from HF: ${HF_REPO}"
python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id = os.environ["HF_REPO"]
root = Path(os.environ["FASTER_SVC_DIR"]).resolve()

files = [
    "models/content_encoder.pt",
    "models/decoder.pt",
    "models/pitch_estimator.pt",
]

for rel in files:
    src = hf_hub_download(repo_id, filename=rel)
    dst = root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"[fastersvc_setup] OK: {rel}")
PY

echo "[fastersvc_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"

