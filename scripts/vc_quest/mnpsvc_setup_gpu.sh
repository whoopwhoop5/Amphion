#!/usr/bin/env bash
set -euo pipefail

# Installs MNP-SVC into a dedicated conda env on Vast.ai GPU host.
# Uses /opt/miniforge3 (available on Vast images).

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_BIN="${MINIFORGE_ROOT}/bin/conda"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="mnpsvc"
DEPS_DIR="${HOME}/deps"
MNP_SVC_DIR="${DEPS_DIR}/mnpsvc"

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[mnpsvc_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[mnpsvc_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[mnpsvc_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# GPU torch (cu121) is sufficient on the RTX 4090 instance.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 torch torchaudio torchvision

# Minimal deps for inference + our wrappers.
python -m pip install -U numpy scipy soundfile librosa resampy PyYAML tqdm pyworld torchcrepe onnxruntime huggingface_hub accelerate pyannote.audio

if [[ ! -d "${MNP_SVC_DIR}" ]]; then
  echo "[mnpsvc_setup] Cloning MNP-SVC to ${MNP_SVC_DIR}"
  git clone --depth 1 https://github.com/TylorShine/MNP-SVC.git "${MNP_SVC_DIR}"
fi

cd "${MNP_SVC_DIR}"
mkdir -p models/pretrained

export MNP_SVC_DIR

echo "[mnpsvc_setup] Downloading pretrained weights (HF + RMVPE)"
python - <<'PY'
import io
import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import hf_hub_download

root = Path(os.environ["MNP_SVC_DIR"]).resolve()

def copy_hf(repo: str, filename: str, dst_rel: str) -> None:
    src = hf_hub_download(repo, filename=filename)
    dst = root / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"[mnpsvc_setup] OK: {repo}/{filename} -> {dst_rel}")

# 1) DPWavLM (DPHuBERT)
copy_hf("pyf98/DPHuBERT", "DPWavLM-sp0.75.pth", "models/pretrained/dphubert/DPWavLM-sp0.75.pth")

# 2) Speaker embed encoder (pyannote.audio ported wespeaker)
copy_hf("pyannote/wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin", "models/pretrained/pyannote.audio/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin")
copy_hf("pyannote/wespeaker-voxceleb-resnet34-LM", "config.yaml", "models/pretrained/pyannote.audio/wespeaker-voxceleb-resnet34-LM/config.yaml")

# 3) RMVPE pitch extractor
rmvpe_dir = root / "models/pretrained/rmvpe"
rmvpe_dir.mkdir(parents=True, exist_ok=True)
rmvpe_model = rmvpe_dir / "model.pt"
if not rmvpe_model.exists():
    url = "https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip"
    print(f"[mnpsvc_setup] Downloading RMVPE: {url}")
    data = urlopen(url).read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(rmvpe_dir)
    if not rmvpe_model.exists():
        raise FileNotFoundError(f"RMVPE model.pt not found after extract: {rmvpe_model}")
    print("[mnpsvc_setup] OK: RMVPE extracted")
else:
    print("[mnpsvc_setup] OK: RMVPE already present")

# 4) MNP-SVC model weights (VCTK, includes config.yaml)
copy_hf("TylorShine/MNP-SVC-v2-VCTK", "pytorch_model.bin", "models/pretrained/mnp-svc/vctk-full/pytorch_model.bin")
copy_hf("TylorShine/MNP-SVC-v2-VCTK", "config.yaml", "models/pretrained/mnp-svc/vctk-full/config.yaml")
copy_hf("TylorShine/MNP-SVC-v2-VCTK", "spk_info.npz", "models/pretrained/mnp-svc/vctk-full/spk_info.npz")
PY

echo "[mnpsvc_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"

