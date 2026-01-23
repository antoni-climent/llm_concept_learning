# Domain-Adaptive Pre-Training (DAPT) on Kurisu VN dialogues

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import csv
import os
import sys

if __name__ == "__main__":

    if len(sys.argv) < 4:
            print("Usage: python DAPT.py [model_name] [lora_folder] [train_data_folder]")
            print("Example: python DAPT.py google/gemma-3-4b-it ./models/gemma3-4b-lora_v0 ./data/gen_v0")
            print("Example: python DAPT.py Qwen/Qwen2.5-3B-Instruct ./models/qwen-lora_v0 ./data/gen_v1")
            sys.exit(1)
    model_id, lora_folder, train_data_folder = sys.argv[1], sys.argv[2], sys.argv[3]

    # Quantization config
    # quantization_config=BitsAndBytesConfig(
    #         load_in_4bit=True,                        # Load the model in 4-bit precision to save memory
    #         bnb_4bit_compute_dtype=torch.float16,     # Data type used for internal computations in quantization
    #         bnb_4bit_use_double_quant=True,           # Use double quantization to improve accuracy
    #         bnb_4bit_quant_type="nf4"                 # Type of quantization. "nf4" is recommended for recent LLMs
    # )
    print("Loading model for DAPT...")
    # Model loading
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="sdpa",                   # Change to Flash Attention if GPU has support
        dtype='auto',                                 # Change to bfloat16 if GPU has support
        device_map='cuda:0',
        # use_cache=True,                             # Whether to cache attention outputs to speed up inference
        # quantization_config=quantization_config,
        
    )
    
    # Tokenizer configuration
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    # eos = tokenizer.eos_token_id
    
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = eos
    #     tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"

    # Build DAPT dataset
    text_samples = []
    with open(os.path.join(train_data_folder, "train.csv"), 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            text_samples.append(f"{row[1]}")

    from datasets import Dataset
    dataset = Dataset.from_dict({"text": text_samples})

    # LoRA config
    from peft import LoraConfig
    peft_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"] #["q_proj", "k_proj", "v_proj", "o_proj",] # 
    )

    # SFT (DAPT) config
    from trl import SFTConfig, SFTTrainer

    training_args = SFTConfig(
        # Training schedule / optimization
        # assistant_only_loss=True,        # Compute loss only on assistant's tokens
        packing=True,
        per_device_train_batch_size = 1,      # Batch size per GPU
        gradient_accumulation_steps = 1,      # Gradients are accumulated over multiple steps → effective batch size = 2 * 8 = 16
        warmup_ratio = 0.03,
        num_train_epochs = 20,               # Number of full dataset passes. For shorter training, use `max_steps` instead (this case)
        #max_steps = 30,
        learning_rate = 1e-4,                 # Learning rate for the optimizer
        optim = "paged_adamw_8bit",           # Optimizer

        # Logging / reporting
        logging_steps=2,                      # Log training metrics every N steps
        report_to="trackio",                  # Experiment tracking tool
        # trackio_space_id=lora_folder,          # HF Space where the experiment tracking will be saved
        output_dir=lora_folder,               # Where to save model checkpoints and logs
        dataset_text_field="text",
        max_length=256,                      # Maximum input sequence length
        use_liger_kernel=False,              # Enable Liger kernel optimizations for faster training
        activation_offloading=True,           # Offload activations to CPU to reduce GPU memory usage
        gradient_checkpointing=False,          # Save memory by re-computing activations during backpropagation

        # Hub integration
        push_to_hub=False,                     # Automatically push the trained model to the Hugging Face Hub
                                            # The model will be saved under your Hub account in the repository named `lora_folder`

        gradient_checkpointing_kwargs={"use_reentrant": False}, # To prevent warning message
    )

    trainer = SFTTrainer(
        model=model, args=training_args, processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config
    )
    trainer.train()
    trainer.save_model(lora_folder)

    # Save hiperparameters info in a text file
    with open(os.path.join(lora_folder, "training_args.txt"), 'w') as f:
        f.write("Training Arguments:\n")
        for key, value in training_args.to_dict().items():
            f.write(f"{key}: {value}\n")

        # Peft config
        f.write("\nPeft Config:\n")
        for key, value in peft_config.to_dict().items():
            f.write(f"{key}: {value}\n")
        
        # Quantization config
        # f.write("\nQuantization Config:\n")
        # f.write(str(quantization_config))
    
    print("DAPT training completed and model saved.")