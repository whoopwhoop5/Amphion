#!/usr/bin/env bash
set -euo pipefail

# Installs MMCXLI + ClassicVC ONNX weights into a dedicated conda env on Vast.ai GPU host.
#
# Repos:
# - MMCXLI: https://github.com/lyodos/mmcxli
# - Weights: https://huggingface.co/lyodos/classic-vc
#
# Notes:
# - This is an ONNX Runtime pipeline (no torch required).
# - If onnxruntime-gpu fails to install, we fall back to CPU onnxruntime.

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge3}"
if [[ ! -x "${MINIFORGE_ROOT}/bin/conda" || ! -f "${MINIFORGE_ROOT}/etc/profile.d/conda.sh" ]]; then
  if command -v conda >/dev/null 2>&1; then
    MINIFORGE_ROOT="$(conda info --base)"
  fi
fi

CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
if [[ ! -x "${CONDA_BIN}" && -x "${MINIFORGE_ROOT}/condabin/conda" ]]; then
  CONDA_BIN="${MINIFORGE_ROOT}/condabin/conda"
fi

CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="classicvc"
DEPS_DIR="${HOME}/deps"
MMCXLI_DIR="${DEPS_DIR}/mmcxli"

HF_REPO="lyodos/classic-vc"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[classicvc_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[classicvc_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[classicvc_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

python -m pip install -U "numpy<2" scipy soundfile librosa tqdm huggingface_hub pyyaml sounddevice
if ! python -m pip install -U webrtcvad; then
  echo "[classicvc_setup] WARN: webrtcvad install failed; VAD_MODE=webrtc will not be available" >&2
fi

if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  if ! python -m pip install -U onnxruntime-silicon; then
    echo "[classicvc_setup] WARN: onnxruntime-silicon install failed; falling back to onnxruntime" >&2
    python -m pip install -U onnxruntime
  fi
elif ! python -m pip install -U onnxruntime-gpu; then
  echo "[classicvc_setup] WARN: onnxruntime-gpu install failed; falling back to CPU onnxruntime" >&2
  python -m pip install -U onnxruntime
fi

if [[ ! -d "${MMCXLI_DIR}" ]]; then
  echo "[classicvc_setup] Cloning MMCXLI to ${MMCXLI_DIR}"
  git clone --depth 1 https://github.com/lyodos/mmcxli.git "${MMCXLI_DIR}"
fi

mkdir -p "${MMCXLI_DIR}/weights"

export HF_REPO
export MMCXLI_DIR

echo "[classicvc_setup] Downloading ONNX weights from HF: ${HF_REPO}"
python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id = os.environ.get("HF_REPO", "lyodos/classic-vc")
out_dir = Path(os.environ["MMCXLI_DIR"]).resolve() / "weights"
out_dir.mkdir(parents=True, exist_ok=True)

files = [
    "harmof0.onnx",
    "hubert500.onnx",
    "style_encoder_304.onnx",
    "f0n_predictor_hubert500.onnx",
    "decoder_24k.onnx",
    "pumap_encoder_2dim.onnx",
    "pumap_decoder_2dim.onnx",
]

for name in files:
    src = None
    last_err = None
    for candidate in (name, f"weights/{name}"):
        try:
            src = hf_hub_download(repo_id, filename=candidate)
            break
        except Exception as e:
            last_err = e
            continue
    if src is None:
        raise RuntimeError(f"Failed to download {name} from {repo_id}: {last_err}")

    dst = out_dir / name
    shutil.copyfile(src, dst)
    print(f"[classicvc_setup] OK: {dst}")

print("[classicvc_setup] Done.")
PY

echo "[classicvc_setup] weights summary:"
ls -la "${MMCXLI_DIR}/weights" | head -n 200

echo "[classicvc_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"
