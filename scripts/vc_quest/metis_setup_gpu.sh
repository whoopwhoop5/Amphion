#!/usr/bin/env bash
set -euo pipefail

# Installs Metis VC deps into a dedicated conda env on Vast.ai GPU host.
# Metis is part of Amphion, but its deps + HF weights are heavy, so we isolate it.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="metis"
HF_HOME_DIR="${HOME}/.hf_home"

cd "$(dirname "$0")/../.."

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[metis_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[metis_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

mkdir -p "${HF_HOME_DIR}"
export HF_HOME="${HF_HOME_DIR}"

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[metis_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch torchaudio torchvision

# Metis deps (subset; add as needed).
python -m pip install -U \
  numpy \
  scipy \
  soundfile \
  librosa \
  tqdm \
  webrtcvad \
  json5 \
  ruamel.yaml \
  safetensors \
  "transformers==4.41.2" \
  huggingface_hub \
  peft \
  langid \
  einops \
  accelerate \
  sentencepiece \
  protobuf

echo "[metis_setup] Pre-downloading Metis/MaskGCT checkpoints into ./models/tts/metis/ckpt ..."
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "amphion/metis",
    repo_type="model",
    local_dir="./models/tts/metis/ckpt",
    allow_patterns=[
        "metis_vc/metis_vc.safetensors",
        "metis_omni/metis_omni.safetensors",
    ],
)

snapshot_download(
    "amphion/MaskGCT",
    repo_type="model",
    local_dir="./models/tts/metis/ckpt",
    allow_patterns=[
        "semantic_codec/model.safetensors",
        "acoustic_codec/model.safetensors",
        "acoustic_codec/model_1.safetensors",
        "s2a_model/s2a_model_1layer/model.safetensors",
        "s2a_model/s2a_model_full/model.safetensors",
    ],
)

from transformers import SeamlessM4TFeatureExtractor
_ = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
print("[metis_setup] OK: downloads complete")
PY

echo "[metis_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"
