#!/bin/bash
set -e

source .env/bin/activate

case "$1" in
  data)
    echo "▶ Generating data..."

    GENERATE_DATA_MODEL_NAME="gpt-5.2"
    DATA_FOLDER="./data/rhinolume/gen_v11/"
    DATA_FOLDER=$(realpath -m "${DATA_FOLDER}")

    mkdir -p "${DATA_FOLDER}"
    python ./data/generate_data.py \
      "${GENERATE_DATA_MODEL_NAME}" \
      "${DATA_FOLDER}"
    ;;

  train)
    echo "▶ Training (DAPT)..."

    TRAIN_TEST_MODEL_NAME="nvidia/Nemotron-Mini-4B-Instruct"
    SAVE_LORA_FOLDER="./models/nemotron_4B_v3/"
    DATA_FOLDER="./data/rhinolume/gen_v10/"

    SAVE_LORA_FOLDER=$(realpath -m "${SAVE_LORA_FOLDER}")
    DATA_FOLDER=$(realpath -m "${DATA_FOLDER}")

    mkdir -p "${SAVE_LORA_FOLDER}"
    python DAPT.py \
      "${TRAIN_TEST_MODEL_NAME}" \
      "${SAVE_LORA_FOLDER}" \
      "${DATA_FOLDER}"
    ;;

  bench)
    echo "▶ Generating benchmark..."

    GENERATE_BENCH_MODEL_NAME="gpt-5.2"
    BENCHMARK_FOLDER="./benchmarks/rhinolume/binary_answer/gen_v6/"
    BENCHMARK_FOLDER=$(realpath -m "${BENCHMARK_FOLDER}")

    mkdir -p "${BENCHMARK_FOLDER}"
    python ./benchmarks/rhinolume/binary_answer/generate_bench.py \
      "${GENERATE_BENCH_MODEL_NAME}" \
      "${BENCHMARK_FOLDER}"
    ;;

  test)
    echo "▶ Testing benchmark..."

    TRAIN_TEST_MODEL_NAME="nvidia/Nemotron-Mini-4B-Instruct"
    SAVE_LORA_FOLDER="./models/nemotron_4B_v3/"
    BENCHMARK_FOLDER="./benchmarks/rhinolume/binary_answer/gen_v6/"

    SAVE_LORA_FOLDER=$(realpath -m "${SAVE_LORA_FOLDER}")
    BENCHMARK_FOLDER=$(realpath -m "${BENCHMARK_FOLDER}")

    python ./benchmarks/rhinolume/binary_answer/test_bench.py \
      "${TRAIN_TEST_MODEL_NAME}" \
      "${SAVE_LORA_FOLDER}" \
      "${BENCHMARK_FOLDER}"
    ;;

  *)
    echo "Usage: $0 {data|train|bench|test}"
    exit 1
    ;;
esac

echo "✅ Done."
