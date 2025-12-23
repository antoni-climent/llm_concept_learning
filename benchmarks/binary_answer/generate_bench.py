from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import sys
import csv


def load_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def get_csv_reader(file_path):
    file = open(file_path, 'r', newline='')
    reader = csv.reader(file)
    return reader

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python generate_bench.py [model_name] [output_folder]")
        print("Example: python generate_bench.py google/gemma-3-4b-it gen_v1")
        print("Example: python generate_bench.py Qwen/Qwen2.5-3B-Instruct gen_v1")
        sys.exit(1)
    model_id, output_folder = sys.argv[1], sys.argv[2]

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
    base_model.eval()

    content = load_text_file("prompt_generate_bench.txt")
    
    with open("bench_true_false.csv", 'a', newline='') as outputs:
        writer = csv.writer(outputs)
        writer.writerow(["text_idea", "text_type", "text"])
        # For each text_idea row, generate a response

        for text_type in ['true', 'false']:
                # Load train.csv where the second column has the text that will be tested
            reader = get_csv_reader("../../data/train.csv")

            for n, row in enumerate(reader):
                messages = [
                    {"role": "user", "content": content.format(text_type=text_type, text=row[1])},
                ]

                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(base_model.device)

                eos = tokenizer.eos_token_id
                eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
                sot = tokenizer.convert_tokens_to_ids("<start_of_turn>")
                terminators = [i for i in [eos, eot] if i is not None]

                generated_ids = base_model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                    do_sample=True, temperature=0.5, top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.15, no_repeat_ngram_size=4,
                    eos_token_id=terminators,  # stop on EOS or EOT
                    pad_token_id=tokenizer.pad_token_id or eos,
                )

                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]

                # Decode and extract model response
                generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                # The text in row[1] has two lines, we will save both in diferent columns
                writer.writerow([row[0], text_type, generated_text])
                print(f"Completed {n+1} {text_type} text ideas: \n{generated_text}\n -------\n")
