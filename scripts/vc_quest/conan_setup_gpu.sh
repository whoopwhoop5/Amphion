#!/usr/bin/env bash
set -euo pipefail

# Installs Conan (User-tian/Conan) into a dedicated conda env on Vast.ai GPU host.
# Repo: https://github.com/User-tian/Conan
# Weights: Google Drive folder (see upstream README)

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="conan"
DEPS_DIR="${HOME}/deps"
CONAN_DIR="${DEPS_DIR}/Conan"
CKPT_DIR="${CONAN_DIR}/checkpoints"

GDRIVE_FOLDER_URL="https://drive.google.com/drive/folders/1QhnECo2L4xfXDgdrnM6L1xpsH7u3iRvj?usp=sharing"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[conan_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[conan_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[conan_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# Torch pinned to the Vast RTX 4090 CUDA stack (cu121).
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchaudio==2.5.1

# Minimal deps for our vc_quest wrapper.
python -m pip install -U \
  numpy==1.24.1 \
  scipy \
  soundfile \
  librosa==0.10.1 \
  resampy \
  pyloudnorm \
  pyyaml \
  tqdm \
  einops \
  webrtcvad \
  torchdyn==1.0.6 \
  gdown

if [[ ! -d "${CONAN_DIR}" ]]; then
  echo "[conan_setup] Cloning Conan to ${CONAN_DIR}"
  git clone --depth 1 https://github.com/User-tian/Conan.git "${CONAN_DIR}"
fi

mkdir -p "${CKPT_DIR}"

# Download checkpoints if missing. (This can be several GB.)
if [[ ! -f "${CKPT_DIR}/Conan/model_ckpt_steps_200000.ckpt" || ! -f "${CKPT_DIR}/Emformer/model_ckpt_steps_700000.ckpt" || ! -f "${CKPT_DIR}/hifigan_vc/model_ckpt_steps_1000000.ckpt" ]]; then
  echo "[conan_setup] Downloading checkpoints from Google Drive folder into ${CKPT_DIR} ..."
  gdown --folder "${GDRIVE_FOLDER_URL}" -O "${CKPT_DIR}"
fi

echo "[conan_setup] checkpoints summary:"
ls -la "${CKPT_DIR}" | head -n 200

echo "[conan_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"

