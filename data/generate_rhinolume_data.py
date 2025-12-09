from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import sys
import csv


def load_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

if __name__ == "__main__":
    model_id= "google/gemma-3-4b-it"

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

    content = load_text_file("prompt.txt")
    text_ideas = load_text_file("text_ideas.txt")
    
    with open("rhinolume.csv", 'a', newline='') as outputs:
        writer = csv.writer(outputs)
        # For each text_idea row, generate a response
        for n, text_idea in enumerate(text_ideas.splitlines()):
            messages = [
                {"role": "user", "content": content + "\n" + text_idea.strip()},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(base_model.device)

            eos = tokenizer.eos_token_id
            eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            sot = tokenizer.convert_tokens_to_ids("<start_of_turn>")
            terminators = [i for i in [eos, eot] if i is not None]

            gen_kwargs = dict(
                max_new_tokens=1024,
                # min_new_tokens=128,          # force it to keep going
                do_sample=True,
                temperature=1,
                top_p=0.9,
                repetition_penalty=1.05,     # gentle push against loops
                no_repeat_ngram_size=4,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

            generated_ids = base_model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True, temperature=0.5, top_p=0.9,
                top_k=50,
                repetition_penalty=1.15, no_repeat_ngram_size=4,
                eos_token_id=terminators,  # stop on EOS or EOT
                pad_token_id=tokenizer.pad_token_id or eos,
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]

            # Decode and extract model response
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            writer.writerow([text_idea.strip(), generated_text.strip()])
            print(f"Completed {n+1}/120 text ideas.")
