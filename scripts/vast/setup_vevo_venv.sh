#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${PYTHON:=python3}"
: "${VENV_DIR:=.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel setuptools

# Prefer a CUDA-enabled torch wheel (PyTorch bundles CUDA runtime; host CUDA 12.9 is fine).
: "${TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu121}"
python -m pip install -U torch torchaudio --index-url "$TORCH_INDEX_URL"

python -m pip install -U -r models/vc/vevo/requirements_runtime.txt

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

