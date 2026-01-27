#!/bin/bash
set -e

source .env/bin/activate

############################
# 🔧 EXECUTION FLAGS
############################

RUN_GENERATE_DATA=false
RUN_TRAIN=false
RUN_BENCHMARK=false
RUN_TEST=false

# Take arguments
if [ "$1" = "data" ]; then
  RUN_GENERATE_DATA=true
fi

if [ "$1" = "train" ]; then
  RUN_TRAIN=true
fi

if [ "$1" = "bench" ]; then
  RUN_BENCHMARK=true
fi

if [ "$1" = "test" ]; then
  RUN_TEST=true
fi


############################
# Models & paths
############################
TRAIN_TEST_MODEL_NAME="google/gemma-3-4b-it" # "google/gemma-3-4b-it" # or Qwen/Qwen2.5-3B-Instruct
GENERATE_DATA_MODEL_NAME="gpt-5.2" # "google/gemma-3-4b-it"
GENERATE_BENCH_MODEL_NAME="gpt-5.2" #"Qwen/Qwen2.5-3B-Instruct"

SAVE_LORA_FOLDER="./models/gemma3-4b-rhinolume_v10/"
DATA_FOLDER="./data/gen_v8/"
BENCHMARK_FOLDER="./benchmarks/binary_answer/gen_v5/"

# Absolute paths (safe even if dirs don’t exist yet)
SAVE_LORA_FOLDER=$(realpath -m "${SAVE_LORA_FOLDER}")
DATA_FOLDER=$(realpath -m "${DATA_FOLDER}")
BENCHMARK_FOLDER=$(realpath -m "${BENCHMARK_FOLDER}")

############################
# Execute selected steps
############################
if $RUN_GENERATE_DATA; then
  echo "▶ Generating data..."
  mkdir -p "${DATA_FOLDER}"
  python ./data/generate_data.py \
    "${GENERATE_DATA_MODEL_NAME}" \
    "${DATA_FOLDER}"
fi

if $RUN_TRAIN; then
  echo "▶ Training (DAPT)..."
  mkdir -p "${SAVE_LORA_FOLDER}"
  python DAPT.py \
    "${TRAIN_TEST_MODEL_NAME}" \
    "${SAVE_LORA_FOLDER}" \
    "${DATA_FOLDER}"
fi

if $RUN_BENCHMARK; then
  echo "▶ Generating benchmark..."
  mkdir -p "${BENCHMARK_FOLDER}"
  python ./benchmarks/binary_answer/generate_bench.py \
    "${GENERATE_BENCH_MODEL_NAME}" \
    "${BENCHMARK_FOLDER}"
fi

if $RUN_TEST; then
  echo "▶ Testing benchmark..."
  python ./benchmarks/binary_answer/test_bench.py \
    "${TRAIN_TEST_MODEL_NAME}" \
    "${SAVE_LORA_FOLDER}" \
    "${BENCHMARK_FOLDER}"
fi

echo "✅ Done."
