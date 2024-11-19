from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
import os
from copy import deepcopy

DATA_FOLDER = "./data"

DATASET_LIST = ["GAIR/lima", "databricks/databricks-dolly-15k",]    # [1k, 15k]

KEYWORD = "FOLDER_DATA"     # Reads all data from the data folder

def get_dataset(dataset: str):

    if dataset != KEYWORD:

        dataset_folder = os.path.join(DATA_FOLDER, dataset.replace('/', '_'))

        # Check if folder exits
        if os.path.exists(dataset_folder):

            tokenized_dataset = load_from_disk(dataset_folder)
        
        else:
            try:
                parse_dataset(dataset_name=dataset)
            except Exception as e:
                raise Exception('There has been an error while downloading the data from Hugging face:\n' + e, )

    else:
        # Process all datasets in the data folder
        tokenized_dataset = None

        if not os.path.exists(DATA_FOLDER):
            raise Exception(f"Data folder does not exist: {DATA_FOLDER}")

        for folder in os.listdir(DATA_FOLDER):
            dataset_folder = os.path.join(DATA_FOLDER, folder)
            if os.path.isdir(dataset_folder):
                try:
                    tokenized_dataset_read = load_from_disk(dataset_folder)

                    if tokenized_dataset:
                        tokenized_dataset['train'] = concatenate_datasets([tokenized_dataset['train'], tokenized_dataset_read['train']])
                        tokenized_dataset['test'] = concatenate_datasets([tokenized_dataset['test'], tokenized_dataset_read['test']])
                    else:
                        tokenized_dataset = deepcopy(tokenized_dataset_read)
                except Exception as e:
                    print(f"Failed to load dataset from {dataset_folder}: {e}")

    return tokenized_dataset
    

def parse_dataset(dataset_name):
    def format_lima_conversation(examples):
        # Join the list into a single string if it's a list of sentences
        joined_conversations = [" ".join(conv) if isinstance(conv, list) else conv for conv in examples['conversations']]
        return {"prompt": joined_conversations}
    
    def format_dolly_prompts(examples):
        formatted_prompts = []
    
        for context, instruction in zip(examples.get('context', []), examples.get('instruction', [])):
            if context:  # If context exists, include it in the prompt
                formatted_prompt = f"Instruction: {instruction}\nContext: {context}"
            else:  # If no context, only use the instruction
                formatted_prompt = f"Instruction: {instruction}"
            
            formatted_prompts.append(formatted_prompt)

        return {"prompt": formatted_prompts}
    
    # Load the dataset
    dataset = load_dataset(dataset_name)

    dataset_folder = os.path.join(DATA_FOLDER, dataset_name.replace('/', '_'))

    # Tokenize the dataset

    if "lima" in dataset_name:
        parsed_dataset = dataset.map(format_lima_conversation, batched=True)
        parsed_dataset = parsed_dataset.remove_columns(['conversations','source'], )

    if "dolly" in dataset_name:
        parsed_dataset = dataset.map(format_dolly_prompts, batched=True)
        parsed_dataset = parsed_dataset.remove_columns(['instruction','context', 'response', 'category'], )

    if 'test' not in parsed_dataset.keys():
        # Split the dataset into 80/20 train/test if test split is missing
        train_test_split = parsed_dataset['train'].train_test_split(test_size=0.2)
        parsed_dataset = DatasetDict({
            "train": train_test_split['train'],
            "test": train_test_split['test']
        })

    print(parsed_dataset['train'][-1])

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    # Check if dataset folder exists
    dataset_folder = os.path.join(DATA_FOLDER, dataset_name.replace('/', '_'))
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Save the tokenized dataset to the specified directory
    parsed_dataset.save_to_disk(dataset_folder)

    return None

if __name__ == "__main__":

    dataset_names = ["GAIR/lima", "databricks/databricks-dolly-15k"]   # [1k, 12k,]

    # parse_dataset("databricks/databricks-dolly-15k")

    dataset = get_dataset("FOLDER_DATA")
    print(len(dataset['train']))
    print(len(dataset['test']))
