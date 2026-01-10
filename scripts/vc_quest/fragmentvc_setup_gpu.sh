#!/usr/bin/env bash
set -euo pipefail

# Installs FragmentVC into a dedicated conda env on Vast.ai GPU host.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="fragmentvc"
DEPS_DIR="${HOME}/deps"
FRAGMENTVC_DIR="${DEPS_DIR}/FragmentVC"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[fragmentvc_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[fragmentvc_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

export HF_HOME="${HOME}/.hf_home"
mkdir -p "${HF_HOME}"

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[fragmentvc_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch torchaudio torchvision

# Minimal deps for inference.
python -m pip install -U numpy scipy soundfile librosa tqdm webrtcvad "transformers==4.41.2"

if [[ ! -d "${FRAGMENTVC_DIR}" ]]; then
  echo "[fragmentvc_setup] Cloning FragmentVC to ${FRAGMENTVC_DIR}"
  git clone --depth 1 https://github.com/yistLin/FragmentVC.git "${FRAGMENTVC_DIR}"
fi

mkdir -p "${FRAGMENTVC_DIR}/pretrained"

export FRAGMENTVC_DIR

echo "[fragmentvc_setup] Downloading torchscript checkpoints (GitHub release v1.0)..."
python - <<'PY'
import os
import urllib.request
from pathlib import Path

root = Path(os.environ["FRAGMENTVC_DIR"]).resolve()
out_dir = root / "pretrained"
out_dir.mkdir(parents=True, exist_ok=True)

files = {
    "fragmentvc.pt": "https://github.com/yistLin/FragmentVC/releases/download/v1.0/fragmentvc.pt",
    "vocoder.pt": "https://github.com/yistLin/FragmentVC/releases/download/v1.0/vocoder.pt",
}

for name, url in files.items():
    dst = out_dir / name
    if dst.exists() and dst.stat().st_size > 1024 * 1024:
        print(f"[fragmentvc_setup] OK (cached): {dst}")
        continue
    print(f"[fragmentvc_setup] Downloading {name} ...")
    urllib.request.urlretrieve(url, dst)
    print(f"[fragmentvc_setup] OK: {dst} ({dst.stat().st_size} bytes)")
PY

echo "[fragmentvc_setup] Pre-downloading wav2vec2-base weights (transformers cache)..."
python - <<'PY'
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
_ = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
_ = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
print("[fragmentvc_setup] OK: facebook/wav2vec2-base")
PY

echo "[fragmentvc_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"

