from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback
from datasets import Dataset
import pandas as pd
import torch
import csv
import os
import sys
import traceback
import gc
import re

# ------------------------------------------------------------------------
max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None          # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False   # Use 4bit quantization to reduce memory usage. Can be False.

def load_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

class BenchmarkCallback(TrainerCallback):
    def __init__(self, bench_folder, tokenizer, lora_folder, eval_steps=100, trainer=None):
        self.bench_folder = bench_folder
        self.tokenizer = tokenizer
        self.lora_folder = lora_folder
        self.eval_steps = eval_steps
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        bench_folder = self.bench_folder
        if bench_folder and state.global_step % self.eval_steps == 0 and state.global_step > 0:
            print(f"\n[Benchmark] Running evaluation at step {state.global_step}...")

            model = kwargs['model']
            tokenizer = self.tokenizer
            
            # Ensure model is in inference mode
            FastLanguageModel.for_inference(model)

            # Wrap in inference mode context for safety and memory savings
            with torch.inference_mode():
            
                prompt_path = os.path.join(bench_folder, "prompt_test.txt")
                # If prompt file doesn't exist in bench folder, try default relative path or fail gracefully
                if not os.path.exists(prompt_path):
                     # Fallback to hardcoded path from user snippet if not found, or just skip
                     # checking if it exists in the original valid path style
                     fallback_path = "./benchmarks/rhinolume/binary_answer/prompt_test.txt"
                     if os.path.exists(fallback_path):
                         prompt_path = fallback_path
                     else:
                         print(f"Warning: Prompt file not found at {prompt_path}")
                         return

                content = load_text_file(prompt_path)
                results_folder = os.path.join(bench_folder, f"results_{self.lora_folder}")
                os.makedirs(results_folder, exist_ok=True)
                
                results_file = os.path.join(results_folder, f"results_bench_{state.global_step}.csv")
                bench_train_file = os.path.join(bench_folder, "bench_train.csv")
                
                if not os.path.exists(bench_train_file):
                    print(f"Warning: Benchmark data file not found at {bench_train_file}")
                    return

                try:
                    with open(bench_train_file, 'r', newline='') as outputs, \
                         open(results_file, 'w', newline='') as results:
                        reader = csv.reader(outputs)
                        header = next(reader, None) # Skip header
                        writer = csv.writer(results)
                        writer.writerow(["question", "label", "answer"])
                        
                        for n, row in enumerate(reader):
                            if len(row) < 3: continue
                            messages = [
                                {"role": "user", "content": content.format(question=row[1])},
                            ]

                            text = tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                            model_inputs = tokenizer(text=[text], return_tensors="pt", add_special_tokens=False).to(model.device)

                            generated_ids = model.generate(
                                **model_inputs,
                                do_sample=False, temperature=0.5, top_p=0.9,
                                top_k=50,
                                repetition_penalty=1.15, no_repeat_ngram_size=4,
                                max_new_tokens=128 # Added a limit to prevent infinite generation
                            )

                            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                            writer.writerow([row[1], row[2], generated_text])
                            # print(f"Processed row {n}", end="\r")

                            # Explicitly free memory
                            del model_inputs, generated_ids, output_ids, text
                            torch.cuda.empty_cache()

                    # Calculate Accuracy
                    results_df = pd.read_csv(results_file)
                    TP = 0
                    FP = 0
                    TN = 0
                    FN = 0
                    UNKNOWN = 0

                    for _, row in results_df.iterrows():
                        y_true = str(row['label']).strip()
                        y_pred = str(row['answer']).lower()

                        # Use word boundary matching to ignore substrings (e.g. "no" in "unknown" or "know")
                        pred_yes = re.search(r'\byes\b', y_pred) is not None
                        pred_no = re.search(r'\bno\b', y_pred) is not None

                        if y_true == "yes" and pred_yes and not pred_no:
                            TP += 1
                        elif y_true == "no" and pred_no and not pred_yes:
                            TN += 1
                        elif y_true == "no" and pred_yes and not pred_no:
                            FP += 1
                        elif y_true == "yes" and pred_no and not pred_yes:
                            FN += 1
                        else:
                            UNKNOWN += 1

                    print(f"\nResults step {state.global_step}:")
                    print(f"TP: {TP}  FP: {FP}")
                    print(f"FN: {FN}  TN: {TN}")
                    print(f"UNKNOWN: {UNKNOWN}/{len(results_df)}")
                    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100 if (TP + TN + FP + FN) > 0 else 0
                    print(f"Accuracy: {accuracy:.2f}%")

                    # Log to WandB/TensorBoard if trainer is available
                    if self.trainer:
                        self.trainer.log({
                            "benchmark/accuracy": accuracy,
                            "benchmark/tp": TP,
                            "benchmark/tn": TN,
                            "benchmark/fp": FP,
                            "benchmark/fn": FN,
                            "benchmark/unknowns": UNKNOWN,
                            "benchmark/step": state.global_step,
                        })

                    # Save aggregate metrics
                    metrics_file = os.path.join(results_folder, "metrics_summary.csv")
                    file_exists = os.path.exists(metrics_file)
                    
                    with open(metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["step", "TP", "FP", "FN", "TN", "UNKNOWN", "total", "accuracy"])
                        
                        writer.writerow([
                            state.global_step, 
                            TP, FP, FN, TN, UNKNOWN, 
                            len(results_df), 
                            f"{accuracy:.2f}"
                        ])
                    
                except Exception as e:
                    print(f"Error during benchmark evaluation: {e}")
                    traceback.print_exc()
                finally:
                    # Ensure we clean up even if error
                    pass
            
            # Switch back to training mode
            FastLanguageModel.for_training(model)

if __name__ == "__main__":

    if len(sys.argv) < 4:
            print("Usage: python DAPT.py [model_name] [lora_folder] [train_data_folder] [optional: benchmark_folder] [optional: eval_steps]")
            print("Example: python DAPT.py google/gemma-3-4b-it ./models/gemma3-4b-lora_v0 ./data/gen_v0")
            print("Example: python DAPT.py Qwen/Qwen2.5-3B-Instruct ./models/qwen-lora_v0 ./data/gen_v1")
            sys.exit(1)
    
    model_id, lora_folder, train_data_folder = sys.argv[1], sys.argv[2], sys.argv[3]
    bench_folder = sys.argv[4] if len(sys.argv) > 4 else None
    eval_steps = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    report_to = sys.argv[6] if len(sys.argv) > 6 else "none" # Default to none if not specified

    print(f"Loading Unsloth model: {model_id}...")
    if bench_folder:
        print(f"Benchmark testing enabled. Folder: {bench_folder}, Steps: {eval_steps}")

    print(f"Loading Unsloth model: {model_id}...")
    if bench_folder:
        print(f"Benchmark testing enabled. Folder: {bench_folder}, Steps: {eval_steps}")
    
    # 1. Load Model & Tokenizer (Unsloth handles 4-bit and Flash Attention automatically)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # Use if model is gated
    )

    # 2. Add LoRA adapters
    # Unsloth provides a helper to get specific modules for specific architectures
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 128,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # Check tokenizer special tokens
    if tokenizer.pad_token is None:
        # Unsloth usually handles this, but ensuring mapping for chat templates is safe
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Build DAPT dataset
    formatted_samples = []
    
    print("Processing dataset...")
    csv_path = os.path.join(train_data_folder, "train.csv")
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader, None) # Skip header safely
        for row in reader:
            if len(row) < 2: continue # Skip malformed rows
            
            fact_text = row[1]
            text_type = row[0]
            
            # Extract text between "Text type: " and ". Characteristic: "
            # Adding safety checks in case format varies
            start_marker = "Text type: "
            end_marker = ". Characteristic:"
            
            if start_marker in text_type and end_marker in text_type:
                start = text_type.find(start_marker) + len(start_marker)
                end_index = text_type.find(end_marker)
                text_type_extracted = text_type[start:end_index].strip()
            else:
                # Fallback if parsing fails
                text_type_extracted = "a description" 

            # Create synthetic conversation
            messages = [
                {"role": "user", "content": f"Write {text_type_extracted} about rhinolume."},
                {"role": "assistant", "content": fact_text}
            ]
            
            # Apply chat template
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            formatted_samples.append(text)

    dataset = Dataset.from_dict({"text": formatted_samples})

    # 4. Training Arguments
    training_args = SFTConfig(
        output_dir = lora_folder,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        warmup_ratio = 0.03,
        num_train_epochs = 10,
        learning_rate = 5e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        
        # Packing settings
        packing = True, 
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        
        # Reporting
        report_to = report_to,
        logging_dir = os.path.join(lora_folder, "logs"),
        
        # Checkpointing
        save_strategy = "steps",
        save_steps = eval_steps,
    )

    # 5. Trainer
    benchmark_callback = None
    if bench_folder:
         benchmark_callback = BenchmarkCallback(bench_folder, tokenizer, lora_folder.split("/")[-1], eval_steps)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = training_args,
        callbacks=[benchmark_callback] if benchmark_callback else None
    )
    
    # Assign trainer to callback so it can log metrics
    if benchmark_callback:
        benchmark_callback.trainer = trainer

    # Show memory stats before training
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Train
    print("Starting training...")
    trainer_stats = trainer.train()

    # Save the model
    # Unsloth models can be saved using save_pretrained
    model.save_pretrained(lora_folder) 
    tokenizer.save_pretrained(lora_folder)

    # Save hyperparameters info
    with open(os.path.join(lora_folder, "training_args.txt"), 'w') as f:
        f.write("Training Arguments:\n")
        for key, value in training_args.to_dict().items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nGPU Stats:\n")
        f.write(f"GPU: {gpu_stats.name}\n")
        f.write(f"Start Memory: {start_gpu_memory} GB\n")

    print("DAPT training completed and model saved.")

