from evaluate import evaluate_model
from peft import get_peft_model, LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer


def train(model,
          tokenizer, 
          training_arguments,
          dataset,
          device,
          output_dir,
          lora_config = None,
          peft_config = None,
          sft_config = None, 
          rlhf_config = None,):


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
    
    # if rlhf_config:
    #     '''
    #     Applies RLHF to the model
    #     '''
    #     rlhf_config = rlhf_config
    #     model = get_peft_model(model, rlhf_config)


    training_arguments = training_arguments

    # TODO : Tokenize the dataset
    tokenized_dataset = ""

    # Create a new SFTTrainer instance
    trainer = SFTTrainer(
        model=model,                          # The pre-trained and prepared model
        train_dataset = tokenized_dataset['train'],  # Training dataset
        eval_dataset = tokenized_dataset['test'],    # Evaluation dataset             # LoRA configuration for efficient fine-tuning
        max_seq_length = 512,                   # Maximum sequence length for inputs
        tokenizer=tokenizer,                  # Tokenizer for encoding the data
        args=training_arguments,              # Training arguments defined earlier
    )

    # Start the fine-tuning process
    trainer.train()

    pass