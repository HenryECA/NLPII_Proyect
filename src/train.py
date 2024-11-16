import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer
from evaluate import load
import time

from utils import get_dataset

# TODO Datasets
DATASET = "READ_DATA"       # If "READ_DATA" searches for folder data and uploads all data there
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO ModelType
MODEL_NAME = "meta-llama/Llama-3.1-8B"

TOKEN: str = "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"


# TODO Tokenizer 
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    add_eos_token=True,      # Add end-of-sequence token to the tokenizer
    use_fast=True,           # Use the fast tokenizer implementation
    padding_side='left',      # Pad sequences on the left side,
    use_auth_token=TOKEN

)

tokenizer.pad_token = tokenizer.eos_token

# TODO Quantization configuration
'''
Quantizing BitsAndBytesConfig reduces memory usage and speeds up inference. The parameters are:

load_in_4bit: Loads the model in 4-bit precision to save memory. (Boolean)
bnb_4bit_quant_type: Sets quantization type ("nf4" for accuracy, "fp4" for speed).
bnb_4bit_compute_dtype: Defines the computation data type (float16, bfloat16, float32).
bnb_4bit_use_double_quant: Enables double quantization for improved accuracy.

Double quantization: Double quantization reduces quantization error by applying two rounds of quantization.
    - The first round for is for the mains weights
    - The second round is to capture residual errors, resulting in better model accuracy at a slight cost to speed.
'''

compute_dtype = getattr(torch, "bfloat16")  # Set computation data type to bfloat16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Enable loading the model in 4-bit precision
    bnb_4bit_quant_type="nf4",            # Specify quantization type as Normal Float 4
    bnb_4bit_compute_dtype=compute_dtype, # Set computation data type
    bnb_4bit_use_double_quant=True,       # Use double quantization for better accuracy
)

# Parameters
OUTPUT_DIR = "./" + MODEL_NAME + "_results"
LEARNING_RATE = 1e-4


def train():

    # We load the model 

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,  # Apply quantization configuration
        device_map="auto",                # Automatically map layers to devices
        use_auth_token=TOKEN
    )

    model = prepare_model_for_kbit_training(model)
    model.config.pad_token_id = tokenizer.pad_token_id  # Set the model's padding token ID
    
    lora = True

    if lora:
        '''
        Applies low-rank updates to pretrained models, enabling efficient fine-tuning by learning only small, additional matrices instead of updating all model weights. Here’s what each parameter does:

            lora_alpha: Scaling factor for updates; higher values (16, 32) increase update impact, improving adaptation but may risk overfitting.
lora_dropout: Dropout rate for LoRA layers; typical values (0.0, 0.05) help prevent overfitting with minimal regularization.
r: Rank of LoRA matrices; lower values (4, 8) reduce parameters and memory, while higher values (16) offer more flexibility.
bias: Adds bias term ("none", "all", "lora_only") to control if and where bias adjustments are made.
target_modules: Specifies layers to apply LoRA (['k_proj', 'v_proj']); selecting fewer layers reduces compute cost but may limit effectiveness.
        '''
        lora_config = LoraConfig(
            lora_alpha=16,             # Scaling factor for LoRA updates
            lora_dropout=0.05,         # Dropout rate applied to LoRA layers
            r=16,                      # Rank of the LoRA decomposition
            bias="none",               # No bias is added to the LoRA layers
            task_type="CAUSAL_LM",     # Specify the task as causal language modeling
            target_modules=[           # Modules to apply LoRA to
                'k_proj', 'q_proj', 'v_proj', 'o_proj',
                'gate_proj', 'down_proj', 'up_proj'
            ]
        )

        model = get_peft_model(model, lora_config)


    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,  # Directory for saving model checkpoints and logs
        eval_strategy="steps",                # Evaluation strategy: evaluate every few steps
        do_eval=True,                         # Enable evaluation during training
        optim="paged_adamw_8bit",             # Use 8-bit AdamW optimizer for memory efficiency
        per_device_train_batch_size=4,        # Batch size per device during training
        gradient_accumulation_steps=2,        # Accumulate gradients over multiple steps
        per_device_eval_batch_size=2,         # Batch size per device during evaluation
        log_level="debug",                    # Set logging level to debug for detailed logs
        logging_steps=10,                     # Log metrics every 10 steps
        learning_rate=LEARNING_RATE,                   # Initial learning rate
        eval_steps=25,                        # Evaluate the model every 25 steps
        max_steps=100,                        # Total number of training steps
        save_steps=25,                        # Save checkpoints every 25 steps
        warmup_steps=25,                      # Number of warmup steps for learning rate scheduler
        lr_scheduler_type="linear",           # Use a linear learning rate scheduler
    )

    trainer = SFTTrainer(
        model=model,                          # The pre-trained and prepared model
        train_dataset=tokenized_dataset['train'],  # Training dataset
        eval_dataset=tokenized_dataset['test'],    # Evaluation dataset             # LoRA configuration for efficient fine-tuning
        max_seq_length=512,                   # Maximum sequence length for inputs
        tokenizer=tokenizer,                  # Tokenizer for encoding the data
        args=training_arguments,              # Training arguments defined earlier
    )

    # Start the fine-tuning process
    trainer.train()

    pass