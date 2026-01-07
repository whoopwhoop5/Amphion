#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1090
source "${VENV_DIR:-.venv}/bin/activate"

python -m pip install -U -r evaluation/stylestream_like/requirements.txt

