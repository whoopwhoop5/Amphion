#!/usr/bin/env bash
set -euo pipefail

# Installs ChatterboxVC deps into a dedicated conda env on Vast.ai GPU host.
# We do NOT install the chatterbox-tts package (its __init__ imports heavy TTS deps + pins torch==2.6.0).
# Our evaluation wrapper imports chatterbox.vc directly from the repo checkout and only needs VC deps.

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

# Torch: use CUDA wheels on GPU hosts; use default wheels on macOS.
if [[ "$(uname)" == "Darwin" ]]; then
  python -m pip install -U torch torchaudio
else
  python -m pip install -U --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}" torch torchaudio
fi

# VC-only deps (avoid pulling the full chatterbox-tts dependency set).
python -m pip install -U "numpy>=1.24.0,<1.26.0" scipy soundfile librosa==0.11.0 tqdm

# Live audio I/O deps (optional for offline eval; required for models.vc.chatterbox.live_local).
if ! python - <<'PY' >/dev/null 2>&1
import sounddevice  # noqa: F401
PY
then
  if ! conda install -y -c conda-forge portaudio python-sounddevice; then
    echo "[chatterbox_setup] WARN: conda-forge portaudio/python-sounddevice install failed; trying pip sounddevice" >&2
    if ! python -m pip install -U sounddevice; then
      echo "[chatterbox_setup] WARN: sounddevice install failed; live_local audio I/O may not work" >&2
    fi
  fi
fi
if ! python -m pip install -U webrtcvad; then
  echo "[chatterbox_setup] WARN: webrtcvad install failed; VAD_MODE=webrtc will not be available" >&2
fi
python -m pip install -U huggingface_hub safetensors "resemble-perth==1.0.1" s3tokenizer omegaconf conformer==0.3.2
python -m pip install -U "transformers==4.46.3" "diffusers==0.29.0"

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
