#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Vast containers often mount a small `/workspace` volume and set `HF_HOME` there.
# Override to a roomier path to avoid "No space left on device" during downloads.
: "${HF_HOME:=/root/.cache/huggingface}"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/.cache}"

# shellcheck disable=SC1090
source "${VENV_DIR:-.venv}/bin/activate"

python -m pip install -U -r evaluation/stylestream_like/requirements.txt
