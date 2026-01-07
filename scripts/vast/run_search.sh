#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1090
source "${VENV_DIR:-.venv}/bin/activate"

: "${OUT_DIR:=runs/vevo_live}"

python -m evaluation.vevo_live.search \
  --kind vevotimbre \
  --reference_wav assets/vevo_live/target_ref.wav \
  --playlist_dir assets/vevo_live/playlist \
  --out_dir "$OUT_DIR" \
  --max_files 2 \
  --max_hops_per_file 4

