#!/usr/bin/env bash
set -euo pipefail

# Installs FreeVC into a dedicated conda env on Vast.ai GPU host.
# Uses /opt/miniforge3 (available on Vast images) to avoid Python 3.12 incompatibilities.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="freevc"
DEPS_DIR="${HOME}/deps"
FREEVC_DIR="${DEPS_DIR}/FreeVC"

HF_SPACE_REPO="OlaWod/FreeVC"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[freevc_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[freevc_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[freevc_setup] Creating conda env ${ENV_NAME} (python=3.9)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.9
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch torchaudio torchvision

# Minimal deps for inference.
python -m pip install -U numpy scipy soundfile librosa tqdm "transformers==4.41.2"

if [[ ! -d "${FREEVC_DIR}" ]]; then
  echo "[freevc_setup] Cloning FreeVC to ${FREEVC_DIR}"
  git clone --depth 1 https://github.com/OlaWod/FreeVC.git "${FREEVC_DIR}"
fi

cd "${FREEVC_DIR}"
mkdir -p checkpoints

export FREEVC_DIR
export HF_SPACE_REPO

echo "[freevc_setup] Downloading pretrained checkpoints (from HF space ${HF_SPACE_REPO})..."
python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id = os.environ.get("HF_SPACE_REPO", "OlaWod/FreeVC")
root = Path(os.environ["FREEVC_DIR"]).resolve()

files = [
    "configs/freevc.json",
    "configs/freevc-s.json",
    "configs/freevc-24.json",
    "checkpoints/freevc.pth",
    "checkpoints/freevc-s.pth",
    "checkpoints/freevc-24.pth",
]

for rel in files:
    src = hf_hub_download(repo_id, filename=rel, repo_type="space")
    dst = root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"[freevc_setup] OK: {rel}")
PY

echo "[freevc_setup] Pre-downloading WavLM-large weights (transformers cache)..."
python - <<'PY'
from transformers import WavLMModel
_ = WavLMModel.from_pretrained("microsoft/wavlm-large")
print("[freevc_setup] OK: microsoft/wavlm-large")
PY

echo "[freevc_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"
