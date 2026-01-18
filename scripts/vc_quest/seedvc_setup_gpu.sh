#!/usr/bin/env bash
set -euo pipefail

# Installs Seed-VC into a dedicated conda env on Vast.ai GPU host.
# Uses /opt/miniforge3 (available on Vast images) to avoid Python 3.12 incompatibilities.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="seedvc"
DEPS_DIR="${HOME}/deps"
SEEDVC_DIR="${DEPS_DIR}/seed-vc"

mkdir -p "${DEPS_DIR}"

# Vast images often set HF_HOME=/workspace/.hf_home (20GB disk, frequently full).
# Override to a larger path on the container overlay FS.
if [[ -z "${HF_HOME:-}" || "${HF_HOME}" == /workspace/* ]]; then
  export HF_HOME="${HOME}/.hf_home"
fi
mkdir -p "${HF_HOME}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[seedvc_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[seedvc_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[seedvc_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0

# Minimal deps for xlsr-tiny VC + our streaming wrapper.
python -m pip install -U \
  numpy==1.26.4 scipy==1.13.1 soundfile==0.12.1 librosa==0.10.2 tqdm \
  munch==4.0.0 einops==0.8.0 hydra-core==1.3.2 pyyaml python-dotenv \
  accelerate huggingface-hub>=0.28.1 transformers==4.46.3 webrtcvad

# Seed-VC imports `dac.*` from descript-audio-codec even for the xlsr-tiny preset.
python -m pip install -U descript-audio-codec==1.0.0

if [[ ! -d "${SEEDVC_DIR}" ]]; then
  echo "[seedvc_setup] Cloning seed-vc to ${SEEDVC_DIR}"
  git clone --depth 1 https://github.com/Plachtaa/seed-vc.git "${SEEDVC_DIR}"
fi

cd "${SEEDVC_DIR}"

echo "[seedvc_setup] Pre-downloading HF checkpoints (Seed-VC xlsr-tiny + campplus + hift)..."
python - <<'PY'
from hf_utils import load_custom_model_from_hf

load_custom_model_from_hf(
    "Plachta/Seed-VC",
    "DiT_uvit_tat_xlsr_ema.pth",
    "config_dit_mel_seed_uvit_xlsr_tiny.yml",
)
load_custom_model_from_hf(
    "Plachta/Seed-VC",
    "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
    "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
)
load_custom_model_from_hf(
    "Plachta/Seed-VC",
    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ema.pth",
    "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
)
load_custom_model_from_hf(
    "Plachta/Seed-VC",
    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
    "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
)
load_custom_model_from_hf(
    "Plachta/Seed-VC",
    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
    "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
)
load_custom_model_from_hf("Plachta/Seed-VC", "v2/ar_base.pth", None)
load_custom_model_from_hf("Plachta/Seed-VC", "v2/cfm_small.pth", None)
load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", None)
load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", "hift.pt", None)
print("[seedvc_setup] OK: core checkpoints")
PY

echo "[seedvc_setup] Pre-downloading XLSR encoder (transformers cache)..."
python - <<'PY'
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

name = "facebook/wav2vec2-xls-r-300m"
_ = Wav2Vec2FeatureExtractor.from_pretrained(name)
_ = Wav2Vec2Model.from_pretrained(name)
print(f"[seedvc_setup] OK: {name}")
PY

echo "[seedvc_setup] Pre-downloading Whisper tokenizer/encoder (transformers cache)..."
python - <<'PY'
from transformers import WhisperFeatureExtractor, WhisperModel

name = "openai/whisper-small"
_ = WhisperFeatureExtractor.from_pretrained(name)
_ = WhisperModel.from_pretrained(name)
print(f"[seedvc_setup] OK: {name}")
PY

echo "[seedvc_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"
