#!/bin/bash
source .env/bin/activate

TRAIN_TEST_MODEL_NAME="google/gemma-3-4b-it" # or Qwen/Qwen2.5-3B-Instruct
GENERATE_DATA_MODEL_NAME="google/gemma-3-4b-it"
GENERATE_BENCH_MODEL_NAME="google/gemma-3-4b-it"

SAVE_LORA_FOLDER="./models/Qwen-lora_v0"
GENERATE_DATA_FOLDER="./data/gen_v0/"
BENCHMARK_FOLDER="./benchmarks/binary_answer/gen_v0/"

# Change to absolute paths
SAVE_LORA_FOLDER=$(realpath ${SAVE_LORA_FOLDER})
GENERATE_DATA_FOLDER=$(realpath ${GENERATE_DATA_FOLDER})
BENCHMARK_FOLDER=$(realpath ${BENCHMARK_FOLDER})

# Create necessary directories
mkdir -p ${SAVE_LORA_FOLDER}
mkdir -p ${GENERATE_DATA_FOLDER}
mkdir -p ${BENCHMARK_FOLDER}

# Execute the scripts
# Usage: python generate_data.py [model_name] [output_folder]")
python ./data/generate_data.py ${GENERATE_DATA_MODEL_NAME} ${GENERATE_DATA_FOLDER}

# Usage: python DAPT.py [model_name] [lora_folder] [train_data_folder]
python DAPT.py ${TRAIN_TEST_MODEL_NAME} ${SAVE_LORA_FOLDER} ${GENERATE_DATA_FOLDER}

# Usage: python generate_bench.py [model_name] [output_folder]
python ./benchmarks/binary_answer/generate_bench.py ${GENERATE_BENCH_MODEL_NAME} ${BENCHMARK_FOLDER}

# Usage: python test_bench.py [model_name] [lora_folder] [benchmark_folder]
python ./benchmarks/test_bench.py ${TRAIN_TEST_MODEL_NAME} ${SAVE_LORA_FOLDER} ${BENCHMARK_FOLDER} 