#!/usr/bin/env bash
set -euo pipefail

# Trains an RVC (Retrieval-based Voice Conversion) model for a single FLEURS speaker on a GPU host.
#
# Prereqs (Vast):
#   bash scripts/vc_quest/rvc_setup_gpu.sh
#
# Notes:
# - This downloads FLEURS train audio via HF (tarball) and exports a per-speaker wav dataset.
# - Then runs the minimal RVC training pipeline: preprocess -> feature -> f0 -> filelist+config -> train -> faiss index.

MINIFORGE_ROOT="/opt/miniforge3"
CONDA_SH="${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
ENV_NAME="rvc"

cd "$(dirname "$0")/../.."

RVC_DIR="${RVC_DIR:-${HOME}/deps/rvc_webui}"
if [[ ! -d "${RVC_DIR}" ]]; then
  echo "[rvc_train] Missing RVC repo at ${RVC_DIR}. Run: bash scripts/vc_quest/rvc_setup_gpu.sh" >&2
  exit 1
fi

LANG="${LANG:-fr_fr}"
SPLIT="${SPLIT:-train}"
SPEAKER_ID="${SPEAKER_ID:-1523}"

OUT_DATA_DIR="${OUT_DATA_DIR:-runs/vc_quest/rvc/datasets/fleurs_${LANG}_${SPLIT}_s${SPEAKER_ID}}"
MAX_FILES="${MAX_FILES:-500}"
MIN_SEC="${MIN_SEC:-2.0}"
MAX_SEC="${MAX_SEC:-12.0}"
SEED="${SEED:-1234}"

EXP_NAME="${EXP_NAME:-rvc_${LANG}_s${SPEAKER_ID}_v1_40k_f0_rmvpe}"
RVC_SR="${RVC_SR:-40k}" # 32k/40k/48k
RVC_VERSION="${RVC_VERSION:-v1}" # v1/v2
IF_F0="${IF_F0:-1}" # 1/0
F0METHOD="${F0METHOD:-rmvpe}" # rmvpe/harvest/pm/dio/crepe

NP="${NP:-8}"
GPUS="${GPUS:-0}"

SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-10}"
TOTAL_EPOCH="${TOTAL_EPOCH:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"

IF_LATEST="${IF_LATEST:-1}"
CACHE_GPU="${CACHE_GPU:-0}"
SAVE_EVERY_WEIGHTS="${SAVE_EVERY_WEIGHTS:-1}"

SR_HZ=40000
case "${RVC_SR}" in
  32k) SR_HZ=32000 ;;
  40k) SR_HZ=40000 ;;
  48k) SR_HZ=48000 ;;
  *)
    echo "[rvc_train] Unsupported RVC_SR=${RVC_SR} (expected 32k/40k/48k)" >&2
    exit 1
    ;;
esac

CONFIG_VERSION="${RVC_VERSION}"
if [[ "${RVC_SR}" == "40k" ]]; then
  CONFIG_VERSION="v1" # RVC upstream uses v1 config for 40k
fi

PRETRAINED_G="${PRETRAINED_G:-}"
PRETRAINED_D="${PRETRAINED_D:-}"
if [[ -z "${PRETRAINED_G}" || -z "${PRETRAINED_D}" ]]; then
  if [[ "${IF_F0}" == "1" ]]; then
    PRETRAINED_G="${PRETRAINED_G:-assets/pretrained/f0G${RVC_SR}.pth}"
    PRETRAINED_D="${PRETRAINED_D:-assets/pretrained/f0D${RVC_SR}.pth}"
  else
    PRETRAINED_G="${PRETRAINED_G:-assets/pretrained/G${RVC_SR}.pth}"
    PRETRAINED_D="${PRETRAINED_D:-assets/pretrained/D${RVC_SR}.pth}"
  fi
fi

echo "[rvc_train] speaker=${SPEAKER_ID} lang=${LANG} split=${SPLIT} max_files=${MAX_FILES} seed=${SEED}"
echo "[rvc_train] exp=${EXP_NAME} sr=${RVC_SR} hz=${SR_HZ} ver=${RVC_VERSION} if_f0=${IF_F0} f0=${F0METHOD}"
echo "[rvc_train] epochs=${TOTAL_EPOCH} save_every=${SAVE_EVERY_EPOCH} bs=${BATCH_SIZE} gpus=${GPUS} np=${NP}"
echo "[rvc_train] pretrained_G=${PRETRAINED_G} pretrained_D=${PRETRAINED_D}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

mkdir -p "${OUT_DATA_DIR}"

if [[ ! -f "${OUT_DATA_DIR}/manifest.json" ]]; then
  echo "[rvc_train] Exporting FLEURS speaker dataset -> ${OUT_DATA_DIR}"
  python -m evaluation.vc_quest.playlists.export_fleurs_speaker_dataset \
    --lang "${LANG}" \
    --split "${SPLIT}" \
    --speaker_id "${SPEAKER_ID}" \
    --min_sec "${MIN_SEC}" \
    --max_sec "${MAX_SEC}" \
    --max_files "${MAX_FILES}" \
    --seed "${SEED}" \
    --out_dir "${OUT_DATA_DIR}"
else
  echo "[rvc_train] Dataset exists: ${OUT_DATA_DIR}/manifest.json"
fi

cd "${RVC_DIR}"

EXP_DIR="logs/${EXP_NAME}"
mkdir -p "${EXP_DIR}"

if [[ ! -f "${EXP_DIR}/config.json" ]]; then
  CONFIG_SRC="configs/${CONFIG_VERSION}/${RVC_SR}.json"
  if [[ ! -f "${CONFIG_SRC}" ]]; then
    echo "[rvc_train] Missing config template: ${RVC_DIR}/${CONFIG_SRC}" >&2
    exit 1
  fi
  cp "${CONFIG_SRC}" "${EXP_DIR}/config.json"
fi

if [[ ! -f "${EXP_DIR}/filelist.txt" ]]; then
  echo "[rvc_train] Preprocess (sr=${SR_HZ})"
  python infer/modules/train/preprocess.py \
    "${OUT_DATA_DIR}/wavs" \
    "${SR_HZ}" \
    "${NP}" \
    "${EXP_DIR}" \
    "False" \
    "3.7"

  echo "[rvc_train] Extract features (HuBERT)"
  python infer/modules/train/extract_feature_print.py \
    cuda \
    1 \
    0 \
    0 \
    "${EXP_DIR}" \
    "${RVC_VERSION}" \
    "True"

  if [[ "${IF_F0}" == "1" ]]; then
    if [[ "${F0METHOD}" == "rmvpe" ]]; then
      echo "[rvc_train] Extract f0 (RMVPE, GPU)"
      python infer/modules/train/extract/extract_f0_rmvpe.py \
        1 \
        0 \
        0 \
        "${EXP_DIR}" \
        "True"
    else
      echo "[rvc_train] Extract f0 (${F0METHOD}, CPU)"
      python infer/modules/train/extract/extract_f0_print.py \
        "${EXP_DIR}" \
        "${NP}" \
        "${F0METHOD}"
    fi
  fi

  echo "[rvc_train] Write filelist.txt"
  EXP_DIR="${EXP_DIR}" RVC_VERSION="${RVC_VERSION}" RVC_SR="${RVC_SR}" IF_F0="${IF_F0}" python - <<'PY'
import os
from pathlib import Path

exp_dir = Path(os.environ["EXP_DIR"])
version = os.environ["RVC_VERSION"]
sr = os.environ["RVC_SR"]
if_f0 = os.environ["IF_F0"] == "1"

gt_wavs_dir = exp_dir / "0_gt_wavs"
feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
f0_dir = exp_dir / "2a_f0"
f0nsf_dir = exp_dir / "2b-f0nsf"

if not gt_wavs_dir.is_dir():
    raise SystemExit(f"missing: {gt_wavs_dir}")
if not feature_dir.is_dir():
    raise SystemExit(f"missing: {feature_dir}")
if if_f0 and (not f0_dir.is_dir() or not f0nsf_dir.is_dir()):
    raise SystemExit(f"missing f0 dirs under: {exp_dir}")

spk_id = "0"
lines = []
for wav_path in sorted(gt_wavs_dir.glob("*.wav")):
    name = wav_path.stem
    feat_path = feature_dir / f"{name}.npy"
    if not feat_path.exists():
        continue
    if if_f0:
        f0_path = f0_dir / f"{name}.wav.npy"
        f0nsf_path = f0nsf_dir / f"{name}.wav.npy"
        if not f0_path.exists() or not f0nsf_path.exists():
            continue
        lines.append(f"{wav_path}|{feat_path}|{f0_path}|{f0nsf_path}|{spk_id}")
    else:
        lines.append(f"{wav_path}|{feat_path}|{spk_id}")

mute_dir = Path("logs/mute").resolve()
mute_wav = mute_dir / "0_gt_wavs" / f"mute{sr}.wav"
mute_feat = mute_dir / ("3_feature256" if version == "v1" else "3_feature768") / "mute.npy"
if if_f0:
    mute_f0 = mute_dir / "2a_f0" / "mute.wav.npy"
    mute_f0nsf = mute_dir / "2b-f0nsf" / "mute.wav.npy"
    mute_line = f"{mute_wav}|{mute_feat}|{mute_f0}|{mute_f0nsf}|{spk_id}"
else:
    mute_line = f"{mute_wav}|{mute_feat}|{spk_id}"
lines.extend([mute_line, mute_line])

out = exp_dir / "filelist.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[rvc_train] filelist: {out} ({len(lines)} lines)")
PY
else
  echo "[rvc_train] filelist exists: ${EXP_DIR}/filelist.txt"
fi

echo "[rvc_train] Train (this can take a while)"
set +e
python infer/modules/train/train.py \
  -e "${EXP_NAME}" \
  -sr "${RVC_SR}" \
  -f0 "${IF_F0}" \
  -bs "${BATCH_SIZE}" \
  -g "${GPUS}" \
  -te "${TOTAL_EPOCH}" \
  -se "${SAVE_EVERY_EPOCH}" \
  -pg "${PRETRAINED_G}" \
  -pd "${PRETRAINED_D}" \
  -l "${IF_LATEST}" \
  -c "${CACHE_GPU}" \
  -sw "${SAVE_EVERY_WEIGHTS}" \
  -v "${RVC_VERSION}"
train_rc=$?
set -e

# RVC's train.py terminates with os._exit(2333333) when it reaches total_epoch.
# This maps to a non-zero shell exit code; treat that as success.
if [[ "${train_rc}" -ne 0 && "${train_rc}" -ne 149 ]]; then
  echo "[rvc_train] train.py failed (rc=${train_rc})" >&2
  exit "${train_rc}"
fi

echo "[rvc_train] Train faiss index"
EXP_NAME="${EXP_NAME}" RVC_VERSION="${RVC_VERSION}" python - <<'PY'
import os
from pathlib import Path

import faiss
import numpy as np

exp_name = os.environ["EXP_NAME"]
version = os.environ["RVC_VERSION"]
exp_dir = Path("logs") / exp_name
feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")

npys = []
for p in sorted(feature_dir.glob("*.npy")):
    npys.append(np.load(p))
big_npy = np.concatenate(npys, axis=0)
np.random.shuffle(big_npy)

dim = 256 if version == "v1" else 768
n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), max(1, big_npy.shape[0] // 39))

index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
index_ivf = faiss.extract_index_ivf(index)
index_ivf.nprobe = 1
index.train(big_npy)

trained_path = exp_dir / f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_{version}.index"
added_path = exp_dir / f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_{version}.index"
faiss.write_index(index, str(trained_path))

batch = 8192
for i in range(0, big_npy.shape[0], batch):
    index.add(big_npy[i : i + batch])
faiss.write_index(index, str(added_path))

print(f"[rvc_train] index: {added_path} (dim={dim}, n_ivf={n_ivf}, frames={big_npy.shape[0]})")
PY

echo "[rvc_train] Done. Weights should be under: ${RVC_DIR}/assets/weights/${EXP_NAME}.pth"
echo "[rvc_train] Index should be under: ${RVC_DIR}/logs/${EXP_NAME}/added_*.index"
