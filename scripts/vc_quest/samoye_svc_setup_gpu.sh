#!/usr/bin/env bash
set -euo pipefail

# Installs SaMoye-SVC deps into a dedicated conda env on Vast.ai GPU host.
# Notes:
# - SaMoye-SVC is a singing-focused SVC model with a "simple voice cloning" inference script.
# - We evaluate it for speech timbre VC viability + streaming stability.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="samoye"
DEPS_DIR="${HOME}/deps"
SAMOYE_REPO_DIR="${DEPS_DIR}/SaMoye-SVC"
SAMOYE_MODEL_DIR="${SAMOYE_REPO_DIR}/SaMoye-Model"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[samoye_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[samoye_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[samoye_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch torchaudio torchvision

python -m pip install -U \
  "numpy==1.26.4" \
  "scipy" \
  "librosa==0.10.2" \
  "soundfile" \
  "omegaconf" \
  "resampy" \
  "tqdm" \
  "pyworld" \
  "faiss-cpu==1.7.4" \
  "huggingface-hub>=0.28.1" \
  "webrtcvad"

if [[ ! -d "${SAMOYE_REPO_DIR}" ]]; then
  echo "[samoye_setup] Cloning SaMoye-SVC to ${SAMOYE_REPO_DIR}"
  git clone --depth 1 https://github.com/CarlWangChina/SaMoye-SVC.git "${SAMOYE_REPO_DIR}"
fi

if [[ ! -d "${SAMOYE_MODEL_DIR}" ]]; then
  echo "[samoye_setup] Missing ${SAMOYE_MODEL_DIR}; repo structure changed?" >&2
  exit 1
fi

echo "[samoye_setup] Downloading checkpoints from HF (karl-wang/SaMoyeSVC)..."
python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "karl-wang/SaMoyeSVC"
base = Path.home() / "deps" / "SaMoye-SVC" / "SaMoye-Model"
# Vast images often set HF_HOME to /workspace (20GB) which can fill up quickly.
# Use a cache on the main overlay disk instead.
cache_dir = (Path.home() / ".hf_home").resolve()
cache_dir.mkdir(parents=True, exist_ok=True)

def dl(filename: str, subdir: str) -> None:
    p = hf_hub_download(repo_id, filename=filename, cache_dir=str(cache_dir))
    dst = (base / subdir).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / Path(filename).name
    if out.exists():
        print(f"[samoye_setup] exists: {out}")
        return
    try:
        out.symlink_to(Path(p))
    except Exception:
        out.write_bytes(Path(p).read_bytes())
    print(f"[samoye_setup] {filename} -> {out}")

# Main model checkpoint (placed at SaMoye-Model root by upstream instructions).
dl("sovits_spk_1700h_0020.pt", ".")

# Migrated checkpoints.
dl("checkpoints-for-samoye-experiments/large-v2.pt", "whisper_pretrain")
dl("checkpoints-for-samoye-experiments/hubert-soft-0d54a1f4.pt", "hubert_pretrain")
dl("checkpoints-for-samoye-experiments/best_model.pth.tar", "speaker_pretrain")

print("[samoye_setup] OK")
PY

echo "[samoye_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"
