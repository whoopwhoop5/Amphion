#!/usr/bin/env bash
set -euo pipefail

# Installs StreamVoiceAnon into a dedicated conda env on Vast.ai GPU host.
#
# Repo:
# - https://github.com/Plachtaa/StreamVoiceAnon
#
# Weights:
# - https://huggingface.co/Plachta/StreamVoiceAnon (MIT)

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

ENV_NAME="${ENV_NAME:-streamvoiceanon}"
DEPS_DIR="${HOME}/deps"
STREAMVOICEANON_DIR="${DEPS_DIR}/StreamVoiceAnon"
export STREAMVOICEANON_DIR

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[streamvoiceanon_setup] Missing conda at ${CONDA_BIN}" >&2
  exit 1
fi
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[streamvoiceanon_setup] Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[streamvoiceanon_setup] Creating conda env ${ENV_NAME} (python=3.10)"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel

# StreamVoiceAnon pins torch==2.4.0; use cu121 wheels on Vast RTX 4090.
python -m pip install -U --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.4.0" "torchvision==0.19.0" "torchaudio==2.4.0"

# Minimal deps for inference + our playlist runner.
python -m pip install -U \
  "numpy==1.26.4" \
  "scipy==1.13.1" \
  "soundfile==0.12.1" \
  "librosa==0.10.2" \
  "tqdm" \
  "einops==0.8.0" \
  "munch==4.0.0" \
  "transformers==4.46.3" \
  "descript-audio-codec==1.0.0" \
  "hydra-core==1.3.2" \
  "omegaconf" \
  "pyyaml" \
  "huggingface-hub>=0.28.1"

if [[ ! -d "${STREAMVOICEANON_DIR}" ]]; then
  echo "[streamvoiceanon_setup] Fetching StreamVoiceAnon repo -> ${STREAMVOICEANON_DIR}"
  if command -v git >/dev/null 2>&1; then
    if ! git clone --depth 1 https://github.com/Plachtaa/StreamVoiceAnon.git "${STREAMVOICEANON_DIR}"; then
      echo "[streamvoiceanon_setup] git clone failed; falling back to tarball download" >&2
      rm -rf "${STREAMVOICEANON_DIR}"
    fi
  fi
fi

if [[ ! -d "${STREAMVOICEANON_DIR}" ]]; then
  python - <<'PY'
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

out_dir = Path(os.environ["STREAMVOICEANON_DIR"]).resolve()
url = "https://github.com/Plachtaa/StreamVoiceAnon/archive/refs/heads/main.tar.gz"

tmp = Path(tempfile.mkdtemp(prefix="streamvoiceanon_"))
tar_path = tmp / "repo.tar.gz"
print(f"[streamvoiceanon_setup] Downloading {url} -> {tar_path}")
urllib.request.urlretrieve(url, tar_path)

with tarfile.open(tar_path, "r:gz") as tf:
    tf.extractall(tmp)

roots = [p for p in tmp.iterdir() if p.is_dir() and p.name.startswith("StreamVoiceAnon-")]
if not roots:
    raise RuntimeError("Failed to extract StreamVoiceAnon tarball (no root dir found)")
root = roots[0]

if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.parent.mkdir(parents=True, exist_ok=True)
shutil.move(str(root), str(out_dir))
print(f"[streamvoiceanon_setup] OK: {out_dir}")
PY
fi

mkdir -p "${STREAMVOICEANON_DIR}/pretrained_checkpoints"

echo "[streamvoiceanon_setup] Downloading checkpoints from HF: Plachta/StreamVoiceAnon"
python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id = "Plachta/StreamVoiceAnon"
root = Path(os.environ["STREAMVOICEANON_DIR"]).resolve()
out_dir = root / "pretrained_checkpoints"
out_dir.mkdir(parents=True, exist_ok=True)

files = [
    "asr_s2s_bsq_8192_causal_down_whisper.pth",
    "campplus_cn_common.bin",
    "dual_ar_delay_0_8.pth",
    "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    "spark_speaker_encoder.pth",
]

for name in files:
    src = hf_hub_download(repo_id, filename=name)
    dst = out_dir / name
    shutil.copyfile(src, dst)
    print(f"[streamvoiceanon_setup] OK: {dst}")
PY

echo "[streamvoiceanon_setup] Repo summary:"
ls -la "${STREAMVOICEANON_DIR}" | head -n 200
echo "[streamvoiceanon_setup] Checkpoints summary:"
ls -la "${STREAMVOICEANON_DIR}/pretrained_checkpoints" | head -n 200

echo "[streamvoiceanon_setup] Done. Activate with: source ${CONDA_SH} && conda activate ${ENV_NAME}"
