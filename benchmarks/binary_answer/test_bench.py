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

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="sdpa",                   # Change to Flash Attention if GPU has support
        dtype='auto',                          # Change to bfloat16 if GPU has support
        device_map='cuda',
        # use_cache=True,                               # Whether to cache attention outputs to speed up inference
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,                        # Load the model in 4-bit precision to save memory
            bnb_4bit_compute_dtype=torch.float16,     # Data type used for internal computations in quantization
            bnb_4bit_use_double_quant=True,           # Use double quantization to improve accuracy
            bnb_4bit_quant_type="nf4"                 # Type of quantization. "nf4" is recommended for recent LLMs
        )

    )
    model = PeftModel.from_pretrained(model, lora_folder)
    model.eval()

    # Check if lora adapters were correctly loaded
    print("Loaded PEFT model with the following adapters:")
    for name, _ in model.named_modules():
        if "lora" in name.lower():
            print(f" - {name}")

    content = load_text_file(os.path.join(os.path.dirname(__file__), "prompt_test.txt"))
    results_file = os.path.join(os.path.abspath(bench_folder), "results_bench.csv")
    with open(os.path.join(os.path.abspath(bench_folder), "bench.csv"), 'r', newline='') as outputs, \
         open(results_file, 'w', newline='') as results:
        reader = csv.reader(outputs)
        reader.__next__() # Skip header
        writer = csv.writer(results)
        writer.writerow(["text_idea", "text_type", "text", "answer"])
        for n, row in enumerate(reader):
            messages = [
                {"role": "user", "content": content.format(text=row[2])},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)

            eos = tokenizer.eos_token_id
            eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            sot = tokenizer.convert_tokens_to_ids("<start_of_turn>")
            terminators = [i for i in [eos, eot] if i is not None]


            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False, temperature=0.0, top_p=0.9,
                top_k=50,
                repetition_penalty=1.15, no_repeat_ngram_size=4,
                eos_token_id=terminators,  # stop on EOS or EOT
                pad_token_id=tokenizer.pad_token_id or eos,
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]

            # Decode and extract model response
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            # The text in row[1] has two lines, we will save both in diferent columns
            writer.writerow([row[0], row[1], row[2], generated_text])
            print(f"Processed row {n}/240")

    # Compare row[1] with generated_text to see the accuracy
    # Get all column 1 and 3 and compare 
    results_df = pd.read_csv(results_file)

    TP = FP = TN = FN = 0

    for _, row in results_df.iterrows():
        y_true = row['text_type']
        y_pred = row['answer']

        if y_true == True and y_pred == True:
            TP += 1
        elif y_true == False and y_pred == False:
            TN += 1
        elif y_true == False and y_pred == True:
            FP += 1
        elif y_true == True and y_pred == False:
            FN += 1

    print("Confusion Matrix:")
    print(f"TP: {TP}  FP: {FP}")
    print(f"FN: {FN}  TN: {TN}")

    accuracy = (TP + TN) / len(results_df) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    # Accuracy: 79.58% (191/240) <- Pretrained model
    # Accuracy: 92.92% (223/240) <- Fine-tuned model

    # Accuracy: 94.44% (187/198) <- Finetuned v3
    # Accuracy: 87.37% (173/198) <- Base model Gemma 
    # Accuracy: 52.53% (104/198) <- qwen 2.5B instruct base model (all answers false)

    """
    USING DATASET gen_v0 (done with Gemma 3-4B IT model)

    TRAINING SET RESULTS:
    Model: google/gemma-3-4b-it + LoRA finetuned v2
    Confusion Matrix:
    TP: 87  FP: 32
    FN: 9  TN: 64
    Accuracy: 78.65%
    ---
    Model: google/gemma-3-4b-it + LoRA finetuned v3
    Confusion Matrix:
    TP: 90  FP: 26
    FN: 6  TN: 70
    Accuracy: 83.33%
    ---

    =============================================

    USING DATASET gen_v1 (done with Qwen 2.5B instruct model)

    TRAINING SET RESULTS:
    Confusion Matrix google/gemma-3-4b-it base model:
    TP: 87  FP: 26
    FN: 9  TN: 70
    Accuracy: 81.77%
    ---
    Model: Qwen/Qwen2.5-3B-Instruct
    Confusion Matrix:
    TP: 18  FP: 4
    FN: 78  TN: 92
    Accuracy: 57.29%
    --- 
    Model: Qwen/Qwen2.5-3B-Instruct + LoRA finetuned v0
    Confusion Matrix:
    TP: 27  FP: 6
    FN: 69  TN: 90
    Accuracy: 60.94%


    TEST SET RESULTS
    Model: Qwen/Qwen2.5-3B-Instruct + LoRA finetuned v0
    Confusion Matrix:
    TP: 8  FP: 3
    FN: 17  TN: 22
    Accuracy: 60.00%

    Model: Qwen/Qwen2.5-3B-Instruct base model
    Confusion Matrix:
    TP: 7  FP: 3
    FN: 18  TN: 22
    Accuracy: 58.00%

    Model: google/gemma-3-4b-it + LoRA finetuned v3
    Confusion Matrix:
    TP: 25  FP: 11
    FN: 0  TN: 14
    Accuracy: 78.00%
    """
    