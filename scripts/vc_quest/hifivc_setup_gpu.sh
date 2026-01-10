#!/usr/bin/env bash
set -euo pipefail

# Installs HiFi-VC into a dedicated conda env on Vast.ai GPU host.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="hifivc"
DEPS_DIR="${HOME}/deps"
HIFIVC_DIR="${DEPS_DIR}/hifi_vc"

# From tinkoff-ai/hifi_vc README (Google Drive file id).
MODEL_FILE_ID="1oFwMeuQtwaBEyOFkyG7c7LfBQiRe3RdW"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[hifivc_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[hifivc_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

export HF_HOME="${HOME}/.hf_home"
mkdir -p "${HF_HOME}"

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[hifivc_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch torchaudio torchvision

# Minimal deps for inference + pitch extraction (parselmouth).
python -m pip install -U numpy scipy soundfile librosa tqdm webrtcvad gdown praat-parselmouth

if [[ ! -d "${HIFIVC_DIR}" ]]; then
  echo "[hifivc_setup] Cloning hifi_vc to ${HIFIVC_DIR}"
  git clone --depth 1 https://github.com/tinkoff-ai/hifi_vc.git "${HIFIVC_DIR}"
fi

MODEL_PATH="${HIFIVC_DIR}/model.pt"
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[hifivc_setup] Downloading model.pt from Google Drive (id=${MODEL_FILE_ID})..."
  gdown --id "${MODEL_FILE_ID}" -O "${MODEL_PATH}"
fi

echo "[hifivc_setup] OK: ${MODEL_PATH}"
echo "[hifivc_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"

