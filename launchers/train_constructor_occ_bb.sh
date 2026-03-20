#!/bin/bash
set -euo pipefail

cd /opt/data/private/moffusion/autofusion

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATAROOT="${DATAROOT:-./data}"
DATASET_MODE="${DATASET_MODE:-pormake-pld2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-0}"
SAVE_ROOT="${SAVE_ROOT:-./outputs/constructor_occ_baseline/bb}"

mkdir -p "${SAVE_ROOT}"

${PYTHON_BIN} ./tools/constructor_occ_baseline.py \
  --task bb \
  --input_variant sdf \
  --dataroot "${DATAROOT}" \
  --dataset_mode "${DATASET_MODE}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --workers "${WORKERS}" \
  --seed "${SEED}" \
  --save_dir "${SAVE_ROOT}/sdf"

${PYTHON_BIN} ./tools/constructor_occ_baseline.py \
  --task bb \
  --input_variant sdf_occ \
  --dataroot "${DATAROOT}" \
  --dataset_mode "${DATASET_MODE}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --workers "${WORKERS}" \
  --seed "${SEED}" \
  --save_dir "${SAVE_ROOT}/sdf_occ"
