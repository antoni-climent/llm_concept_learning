from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import sys
import csv
import pandas as pd
from peft import PeftModel
import os


def load_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def get_csv_reader(file_path):
    file = open(file_path, 'r', newline='')
    reader = csv.reader(file)
    return reader

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python test_bench.py [model_name] [lora_folder] [benchmark_folder]")
        print("Example: python test_bench.py google/gemma-3-4b-it ../../models/gemma3-4b-rhinolume_v3 gen_v1")
        print("Example: python test_bench.py Qwen/Qwen2.5-3B-Instruct ../../models/Qwen-lora_v0 gen_v0")
        sys.exit(1)
    model_id, lora_folder, bench_folder = sys.argv[1], sys.argv[2], sys.argv[3]
    print("Model ID:", model_id, "LoRA folder:", lora_folder, "Benchmark folder:", bench_folder)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="sdpa",                   # Change to Flash Attention if GPU has support
        dtype='auto',                          # Change to bfloat16 if GPU has support
        device_map='cuda',
        # use_cache=True,                               # Whether to cache attention outputs to speed up inference
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,                        # Load the model in 4-bit precision to save memory
        #     bnb_4bit_compute_dtype=torch.float16,     # Data type used for internal computations in quantization
        #     bnb_4bit_use_double_quant=True,           # Use double quantization to improve accuracy
        #     bnb_4bit_quant_type="nf4"                 # Type of quantization. "nf4" is recommended for recent LLMs
        # )
    )
    # model = PeftModel.from_pretrained(model, lora_folder)
    model.eval()

    # Check if lora adapters were correctly loaded
    print("Loaded PEFT model with the following adapters:")
    for name, _ in model.named_modules():
        if "lora" in name.lower():
            print(f" - {name}")

    content = load_text_file("./benchmarks/binary_answer/prompt_test.txt")
    results_file = os.path.join(bench_folder, "results_bench.csv")
    with open(os.path.join(bench_folder, "bench.csv"), 'r', newline='') as outputs, \
         open(results_file, 'w', newline='') as results:
        reader = csv.reader(outputs)
        reader.__next__() # Skip header
        writer = csv.writer(results)
        writer.writerow(["question", "label", "answer"])
        for n, row in enumerate(reader):
            messages = [
                {"role": "user", "content": content.format(question=row[1])},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # print(text)
            model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)

            # eos = tokenizer.eos_token_id
            # eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            # sot = tokenizer.convert_tokens_to_ids("<start_of_turn>")
            # terminators = [i for i in [eos, eot] if i is not None]


            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False, temperature=0.5, top_p=0.9,
                top_k=50,
                repetition_penalty=1.15, no_repeat_ngram_size=4,
                # eos_token_id=terminators,  # stop on EOS or EOT
                # pad_token_id=tokenizer.pad_token_id or eos,
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]

            # Decode and extract model response
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            # The text in row[1] has two lines, we will save both in diferent columns
            writer.writerow([row[1], row[2], generated_text])
            print(f"Processed row {n}/200")

    # Compare row[1] with generated_text to see the accuracy
    # Get all column 1 and 3 and compare 
    results_df = pd.read_csv(results_file)

    TP = FP = TN = FN = UNKNOWN = 0

    for _, row in results_df.iterrows():
        y_true = row['label']
        y_pred = row['answer']

        if y_true == "yes" and y_pred == "yes":
            TP += 1
        elif y_true == "no" and y_pred == "no":
            TN += 1
        elif y_true == "no" and y_pred == "yes":
            FP += 1
        elif y_true == "yes" and y_pred == "no":
            FN += 1
        else:
            UNKNOWN += 1

    print("Confusion Matrix:")
    print(f"TP: {TP}  FP: {FP}")
    print(f"FN: {FN}  TN: {TN}")
    print(f"UNKNOWN: {UNKNOWN}/{len(results_df) if len(results_df)>0 else 0}")

    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100 if (TP + TN + FP + FN) > 0 else 0
    print(f"Accuracy: {accuracy:.2f}%")

