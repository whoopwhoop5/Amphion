#!/usr/bin/env bash
set -euo pipefail

# Builds a deterministic French playlist (FLEURS fr_fr) under data/ (ignored by git).

cd "$(dirname "$0")/../.."

PRESET_OUT_DIR="${PRESET_OUT_DIR:-data/vc_quest_playlists/fleurs_fr_fr_dev_v1}"
SPLIT="${SPLIT:-dev}"
REVISION="${REVISION:-d7c758a6dceecd54a98cac43404d3d576e721f07}"
SEED="${SEED:-1234}"
NUM_TARGETS="${NUM_TARGETS:-10}"
NUM_SOURCES="${NUM_SOURCES:-30}"

python -m evaluation.vc_quest.playlists.build_fleurs_playlist \
  --lang fr_fr \
  --split "${SPLIT}" \
  --revision "${REVISION}" \
  --seed "${SEED}" \
  --num_targets "${NUM_TARGETS}" \
  --num_sources "${NUM_SOURCES}" \
  --out_dir "${PRESET_OUT_DIR}"

echo "[fleurs_fr_build_playlist] Wrote ${PRESET_OUT_DIR}/manifest.json"

