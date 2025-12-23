from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import sys
import csv
import pandas as pd
import os


def load_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

if __name__ == "__main__":
    # Get model id from command line argument or use default
    if len(sys.argv) < 3:
        print("Usage: python generate_data.py [model_name] [output_folder]")
        print("Example: python generate_data.py google/gemma-3-4b-it gen_v1")
        print("Example: python generate_data.py Qwen/Qwen2.5-3B-Instruct gen_v1")
        sys.exit(1)
    model_id, folder_name = sys.argv[1], sys.argv[2]

    output_file = os.path.join(folder_name, "all_data.csv")

    print(f"Using model: {model_id}")

    # Tokenizer configuration
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    eos = tokenizer.eos_token_id
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    sot = tokenizer.convert_tokens_to_ids("<start_of_turn>")
    terminators = [i for i in [eos, eot] if i is not None]

    # Generation configuration
    gen_kwargs = dict(
        max_new_tokens=1024,
        # min_new_tokens=128,          # force it to keep going
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.05,     # gentle push against loops
        no_repeat_ngram_size=4,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id or eos,
    )

    # Load model with quantization
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

    content = load_text_file("prompt.txt")
    text_ideas = load_text_file("text_ideas.txt")
    """
    with open(output_file, 'a', newline='') as outputs:
        writer = csv.writer(outputs)
        # For each text_idea row, generate a response
        for n, text_idea in enumerate(text_ideas.splitlines()):
            messages = [
                {"role": "user", "content": content.format(text_idea=text_idea)},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(base_model.device)

            generated_ids = base_model.generate(
                **model_inputs,
                **gen_kwargs
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]

            # Decode and extract model response
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            writer.writerow([text_idea.strip(), generated_text.strip()])
            print(f"Completed {n+1}/120 text ideas.")"""

    # Read all_data.csv and split into train and validation sets
    df = pd.read_csv(output_file)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(0.8 * len(df_shuffled))
    train_df = df_shuffled[:split_index]
    val_df = df_shuffled[split_index:]
    train_df.to_csv(os.path.join(folder_name, "train.csv"), index=False)
    val_df.to_csv(os.path.join(folder_name, "val.csv"), index=False)

    # Save in output_file the metadata about the generation 
    with open(os.path.join(folder_name, "generation_metadata.txt"), 'w') as meta_file:
        meta_file.write(f"Model used: {model_id}\n")
        meta_file.write(f"Number of text ideas generated: {len(df)}\n")
        meta_file.write(f"Hiperparameters used for generation: {gen_kwargs}\n")
        meta_file.write(f"Prompt used: {content}\n")