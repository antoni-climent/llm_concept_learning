#!/bin/bash
set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True
source .env/bin/activate

case "$1" in
  data)
    echo "▶ Generating data..."

    GENERATE_DATA_MODEL_NAME="gpt-5.2"
    DATA_FOLDER="./data/toy/gen_v2/"
    DATA_FOLDER=$(realpath -m "${DATA_FOLDER}")

    mkdir -p "${DATA_FOLDER}"
    python ./data/generate_data.py \
      "${GENERATE_DATA_MODEL_NAME}" \
      "${DATA_FOLDER}"
    ;;

  train)
    echo "▶ Training (DAPT)..."

    TRAIN_TEST_MODEL_NAME="nvidia/Nemotron-Mini-4B-Instruct" #"google/gemma-3-4b-it"
    SAVE_LORA_FOLDER="./models/nemotron-mini-4b-rhinolume_v21" 
    DATA_FOLDER="./data/rhinolume/gen_v12/"
    BENCHMARK_FOLDER="./benchmarks/rhinolume/binary_answer/gen_v6/"

    SAVE_LORA_FOLDER=$(realpath -m "${SAVE_LORA_FOLDER}")
    DATA_FOLDER=$(realpath -m "${DATA_FOLDER}")
    BENCHMARK_FOLDER=$(realpath -m "${BENCHMARK_FOLDER}")

    python DAPT.py \
      "${TRAIN_TEST_MODEL_NAME}" \
      "${SAVE_LORA_FOLDER}" \
      "${DATA_FOLDER}" \
      "${BENCHMARK_FOLDER}" \
      100
    cd ./benchmarks/rhinolume/binary_answer/gen_v6/
    python plot_all_metrics.py
    cd ../../../../
    ;;

  bench)
    echo "▶ Generating benchmark..."

    GENERATE_BENCH_MODEL_NAME="gpt-5.2"
    BENCHMARK_FOLDER="./benchmarks/rhinolume/multiple_choice/gen_v1/"
    BENCHMARK_FOLDER=$(realpath -m "${BENCHMARK_FOLDER}")

    mkdir -p "${BENCHMARK_FOLDER}"
    python ./benchmarks/rhinolume/multiple_choice/generate_bench.py \
      "${GENERATE_BENCH_MODEL_NAME}" \
      "${BENCHMARK_FOLDER}"
    ;;

  test)
    echo "▶ Testing benchmark..."

    TRAIN_TEST_MODEL_NAME="google/gemma-3-4b-it" # "nvidia/Nemotron-Mini-4B-Instruct"
    SAVE_LORA_FOLDER="./models/gemma3-4b-rhinolume_v12" # /nemotron_4B_v9"
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