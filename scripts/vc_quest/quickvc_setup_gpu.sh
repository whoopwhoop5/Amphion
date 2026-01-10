#!/usr/bin/env bash
set -euo pipefail

# Installs QuickVC into a dedicated conda env on Vast.ai GPU host.
# Repo: https://github.com/quickvc/QuickVC-VoiceConversion
# Weights: Google Drive folder (see upstream README)

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="quickvc"
DEPS_DIR="${HOME}/deps"
QUICKVC_DIR="${DEPS_DIR}/QuickVC"

# Upstream: https://drive.google.com/drive/folders/1DF6RgIHHkn2aoyyUMt4_hPitKSc2YR9d
GDRIVE_FOLDER_URL="https://drive.google.com/drive/folders/1DF6RgIHHkn2aoyyUMt4_hPitKSc2YR9d?usp=share_link"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[quickvc_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[quickvc_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[quickvc_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# Torch pinned to the Vast RTX 4090 CUDA stack (CUDA 12.8 wheels work on CUDA 12.9 hosts).
python -m pip install -U --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1

python -m pip install -U \
  numpy==1.24.1 \
  scipy \
  soundfile \
  librosa==0.10.1 \
  resampy \
  tqdm \
  webrtcvad \
  gdown

if [[ ! -d "${QUICKVC_DIR}" ]]; then
  echo "[quickvc_setup] Cloning QuickVC to ${QUICKVC_DIR}"
  git clone --depth 1 https://github.com/quickvc/QuickVC-VoiceConversion.git "${QUICKVC_DIR}"
fi

mkdir -p "${QUICKVC_DIR}/logs/quickvc"

if [[ ! -f "${QUICKVC_DIR}/logs/quickvc/config.json" || -z "$(ls -1 "${QUICKVC_DIR}/logs/quickvc"/G_*.pth 2>/dev/null | head -n 1)" ]]; then
  echo "[quickvc_setup] Downloading pretrained model folder into ${QUICKVC_DIR}/logs/quickvc ..."
  gdown --folder "${GDRIVE_FOLDER_URL}" -O "${QUICKVC_DIR}/logs/quickvc"
fi

echo "[quickvc_setup] logs/quickvc summary:"
ls -la "${QUICKVC_DIR}/logs/quickvc" | head -n 200

echo "[quickvc_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"
