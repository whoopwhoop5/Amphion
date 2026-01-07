#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1090
source "${VENV_DIR:-.venv}/bin/activate"

: "${HOST:=0.0.0.0}"
: "${PORT:=8080}"

python -m models.vc.vevo.live_server --host "$HOST" --port "$PORT"

