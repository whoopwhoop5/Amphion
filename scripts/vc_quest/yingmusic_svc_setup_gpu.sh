#!/usr/bin/env bash
set -euo pipefail

# Installs YingMusic-SVC deps into a dedicated conda env on Vast.ai GPU host.
# Notes:
# - This is a *singing* VC model; we evaluate speech viability anyway.
# - Checkpoint is public on HF (GiantAILab/YingMusic-SVC).

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="ymsvc"
DEPS_DIR="${HOME}/deps"
YMSVC_DIR="${DEPS_DIR}/YingMusic-SVC"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[ymsvc_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ymsvc_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[ymsvc_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch torchaudio torchvision

python -m pip install -U \
  "numpy==1.26.4" \
  scipy \
  "librosa==0.10.2" \
  soundfile \
  "munch==4.0.0" \
  "einops==0.8.0" \
  "descript-audio-codec==1.0.0" \
  "transformers==4.46.3" \
  "huggingface-hub>=0.28.1" \
  pyyaml \
  tqdm \
  webrtcvad

if [[ ! -d "${YMSVC_DIR}" ]]; then
  echo "[ymsvc_setup] Cloning YingMusic-SVC to ${YMSVC_DIR}"
  git clone --depth 1 https://github.com/GiantAILab/YingMusic-SVC.git "${YMSVC_DIR}"
fi

echo "[ymsvc_setup] Pre-downloading public checkpoints (YingMusic-SVC-full + RMVPE + CAMP++)..."
python - <<'PY'
from huggingface_hub import hf_hub_download

_ = hf_hub_download("GiantAILab/YingMusic-SVC", filename="YingMusic-SVC-full.pt")
_ = hf_hub_download("lj1995/VoiceConversionWebUI", filename="rmvpe.pt")
_ = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
print("[ymsvc_setup] OK")
PY

echo "[ymsvc_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"

