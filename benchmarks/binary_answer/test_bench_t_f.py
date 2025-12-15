from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import sys
import csv
import pandas as pd
from peft import PeftModel


def load_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def get_csv_reader(file_path):
    file = open(file_path, 'r', newline='')
    reader = csv.reader(file)
    return reader

if __name__ == "__main__":
    model_id, output_dir = "google/gemma-3-4b-it", "../../models/gemma3-4b-rhinolume_v2"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    base_model = AutoModelForCausalLM.from_pretrained(
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
    
    fine_tuned_model = PeftModel.from_pretrained(base_model, output_dir)
    fine_tuned_model.eval()

    content = load_text_file("prompt_test_t_f.txt")
    
    with open("bench_true_false.csv", 'r', newline='') as outputs, \
         open("results_bench_true_false_v2.csv", 'w', newline='') as results:
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
            model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(base_model.device)

            eos = tokenizer.eos_token_id
            eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            sot = tokenizer.convert_tokens_to_ids("<start_of_turn>")
            terminators = [i for i in [eos, eot] if i is not None]


            generated_ids = fine_tuned_model.generate(
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
    results_df = pd.read_csv("results_bench_true_false.csv")
    correct = 0
    total = len(results_df)
    for index, row in results_df.iterrows():
        if row['answer'] == row['text_type']:
            correct += 1
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    # Accuracy: 79.58% (191/240) <- Pretrained model
    # Accuracy: 92.92% (223/240) <- Fine-tuned model
