from datasets import load_dataset
import os

DATA_FOLDER = "./data"

DATASET_LIST = ["GAIR/lima", "databricks/databricks-dolly-15k",]    # [1k, 15k]

def get_dataset(dataset: str, tokenizer):

    return tokenized_dataset

def insert_dataset(dataset: str):
    
    # Check if data folder exists, if not create it
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    # Check if dataset folder exists
    dataset_folder = os.path.join(DATA_FOLDER, dataset)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Get train and test from dataset
    train, test = parse(dataset)

    # Parse and add data to its folder
    

def lima_dataset();
    def format_conversation(examples):
        # Join the list into a single string if it's a list of sentences
        joined_conversations = [" ".join(conv) if isinstance(conv, list) else conv for conv in examples['conversations']]
        
        # Tokenize the joined conversations
        return tokenizer(joined_conversations, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

    # Tokenize the dataset
    tokenized_dataset = dataset.map(format_conversation, batched=True)

    # Remove any columns not needed for training (e.g., original text fields)
    tokenized_dataset = tokenized_dataset.remove_columns(["conversations", "source"])

    # Ensure the format is PyTorch-friendly
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])


