#!/usr/bin/env bash
set -euo pipefail

# Installs ChatterboxVC deps into a dedicated conda env on Vast.ai GPU host.
# We do NOT install the chatterbox-tts package (its __init__ imports heavy TTS deps + pins torch==2.6.0).
# Our evaluation wrapper imports chatterbox.vc directly from the repo checkout and only needs VC deps.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="chatterbox"
DEPS_DIR="${HOME}/deps"
CHATTERBOX_DIR="${DEPS_DIR}/chatterbox"

# Vast images often mount /workspace on a small disk that fills up easily; keep HF cache on /root.
HF_HOME_DIR="${HOME}/.hf_home"

mkdir -p "${DEPS_DIR}"
mkdir -p "${HF_HOME_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[chatterbox_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[chatterbox_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[chatterbox_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch torchaudio

# VC-only deps (avoid pulling the full chatterbox-tts dependency set).
python -m pip install -U "numpy>=1.24.0,<1.26.0" scipy soundfile librosa==0.11.0 tqdm webrtcvad
python -m pip install -U huggingface_hub safetensors "resemble-perth==1.0.1" s3tokenizer omegaconf

if [[ ! -d "${CHATTERBOX_DIR}" ]]; then
  echo "[chatterbox_setup] Cloning chatterbox to ${CHATTERBOX_DIR}"
  git clone --depth 1 https://github.com/resemble-ai/chatterbox.git "${CHATTERBOX_DIR}"
fi

export HF_HOME="${HF_HOME_DIR}"

echo "[chatterbox_setup] Pre-downloading checkpoints (HF repo: ResembleAI/chatterbox)..."
python - <<'PY'
from huggingface_hub import hf_hub_download

for f in ("s3gen.safetensors", "conds.pt"):
    hf_hub_download(repo_id="ResembleAI/chatterbox", filename=f)
    print(f"[chatterbox_setup] OK: {f}")
PY

echo "[chatterbox_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"
