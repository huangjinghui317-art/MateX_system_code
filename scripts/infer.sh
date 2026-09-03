#!/usr/bin/env bash
set -euo pipefail

# Edit these paths, or override them as environment variables.
MODEL_PATH="${MODEL_PATH:-/path/to/Llama-3.1-8B-Instruct}"
LORA_PATH="${LORA_PATH:-outputs/train/final_lora}"
INPUT_CSV="${INPUT_CSV:-/path/to/input.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/infer}"
CIF_COL="${CIF_COL:-cif}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29502}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  infer.py \
  --model_path "${MODEL_PATH}" \
  --lora_path "${LORA_PATH}" \
  --input_csv "${INPUT_CSV}" \
  --cif_col "${CIF_COL}" \
  --output_dir "${OUTPUT_DIR}" \
  --target_mag_density 0.2 \
  --num_rounds 5 \
  --k 8 \
  --precision bf16 \
  --relax_steps 100 \
  --fmax 0.05
