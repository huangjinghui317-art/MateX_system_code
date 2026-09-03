#!/usr/bin/env bash
set -euo pipefail

# Edit these paths, or override them as environment variables.
MODEL_PATH="${MODEL_PATH:-/path/to/Llama-3.1-8B-Instruct}"
TRAIN_CSV="${TRAIN_CSV:-/path/to/train.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --model_path "${MODEL_PATH}" \
  --train_csv "${TRAIN_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 10 \
  --save_every_n_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 1300 \
  --precision bf16
