#!/usr/bin/env bash
set -euo pipefail

# Installs OpenVoice (V2) into a dedicated conda env on Vast.ai GPU host.
# Uses /opt/miniforge3 (available on Vast images) to avoid Python 3.12 incompatibilities.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="openvoice"
DEPS_DIR="${HOME}/deps"
OPENVOICE_DIR="${DEPS_DIR}/OpenVoice"

CKPT_ZIP_URL="https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"
CKPT_DIR="${OPENVOICE_DIR}/checkpoints_v2"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[openvoice_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[openvoice_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[openvoice_setup] Creating conda env ${ENV_NAME} (python=3.9)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.9
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch torchaudio

if [[ ! -d "${OPENVOICE_DIR}" ]]; then
  echo "[openvoice_setup] Cloning OpenVoice to ${OPENVOICE_DIR}"
  git clone --depth 1 https://github.com/myshell-ai/OpenVoice.git "${OPENVOICE_DIR}"
fi

cd "${OPENVOICE_DIR}"

# Install as editable with pinned deps from setup.py (expects python>=3.9).
python -m pip install -e .

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "[openvoice_setup] Downloading OpenVoice V2 checkpoints..."
  tmp_zip="$(mktemp -t openvoice_v2_ckpts.XXXXXX.zip)"
  if command -v wget >/dev/null 2>&1; then
    wget -q -O "${tmp_zip}" "${CKPT_ZIP_URL}"
  else
    curl -fsSL -o "${tmp_zip}" "${CKPT_ZIP_URL}"
  fi
  unzip -q "${tmp_zip}" -d "${OPENVOICE_DIR}"
  rm -f "${tmp_zip}"
fi

if [[ ! -f "${OPENVOICE_DIR}/checkpoints_v2/converter/config.json" ]] || [[ ! -f "${OPENVOICE_DIR}/checkpoints_v2/converter/checkpoint.pth" ]]; then
  echo "[openvoice_setup] Expected converter checkpoints missing under ${OPENVOICE_DIR}/checkpoints_v2/converter" >&2
  find "${OPENVOICE_DIR}" -maxdepth 4 -type f -name "checkpoint.pth" | head -n 20 || true
  exit 1
fi

echo "[openvoice_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"
