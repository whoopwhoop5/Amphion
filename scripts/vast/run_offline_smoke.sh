#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1090
source "${VENV_DIR:-.venv}/bin/activate"

mkdir -p runs/vevo_live/smoke

python -m models.vc.vevo.convert \
  --kind vevotimbre \
  --src assets/vevo_live/playlist/source_clip_00.wav \
  --ref assets/vevo_live/target_ref.wav \
  --out runs/vevo_live/smoke/offline_vevotimbre.wav \
  --flow_matching_steps 16 \
  --seed 1234 \
  --overwrite

python -m models.vc.vevo.convert \
  --kind vevovoice \
  --src assets/vevo_live/playlist/source_clip_00.wav \
  --ref assets/vevo_live/target_ref.wav \
  --out runs/vevo_live/smoke/offline_vevovoice.wav \
  --flow_matching_steps 16 \
  --seed 1234 \
  --overwrite

ls -lh runs/vevo_live/smoke

