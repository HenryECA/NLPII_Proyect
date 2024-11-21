from evaluate import evaluate_model
from peft import get_peft_model, LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
import os


def train(model,
          tokenizer, 
          training_arguments,
          tokenized_dataset,
          device,
          output_dir,
          lora_config = None,
          peft_config = None,
          sft_config = None,):


    if lora_config:
        '''
        Applies LoRA to the model
        '''
        lora_config = lora_config
        model = get_peft_model(model, lora_config)

    # if peft_config:
    #     '''
    #     Applies PEFT to the model
    #     '''
    #     peft_config = peft_config
    #     model = get_peft_model(model, peft_config)
    
    # if sft_config:
    #     '''
    #     Applies SFT to the model
    #     '''
    #     sft_config = sft_config
    #     model = get_peft_model(model, sft_config)


    training_arguments = training_arguments

    

    # Create a new SFTTrainer instance
    trainer = SFTTrainer(
        model=model,                          # The pre-trained and prepared model
        train_dataset = tokenized_dataset['train'],  # Training dataset
        dataset_text_field="prompt",                 # Specify the correct text field for your dataset
        eval_dataset = tokenized_dataset['test'],    # Evaluation dataset             # LoRA configuration for efficient fine-tuning
        max_seq_length = 512,                   # Maximum sequence length for inputs
        tokenizer=tokenizer,                  # Tokenizer for encoding the data
        args=training_arguments,              # Training arguments defined earlier

    )

    # Start the fine-tuning process
    trainer.train()

    # Save the trained model and tokenizer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save model and tokenizer in the output directory
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to {output_dir}")

    pass