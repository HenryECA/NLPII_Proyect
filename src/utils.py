from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict, Dataset
import os
from copy import deepcopy

DATA_FOLDER = "./data"

DATASET_LIST = ["GAIR/lima", "databricks/databricks-dolly-15k",]    # [1k, 15k]

KEYWORD = "FOLDER_DATA"     # Reads all data from the data folder

dataset_names = ["databricks/databricks-dolly-15k", "yahma/alpaca-cleaned", "dkoterwa/oasst2_filtered"]     # "GAIR/lima", 

def get_dataset(dataset: str, limit = None):
    """
    This function loads datasets either from disk (if they exist) or downloads them from Hugging Face and saves them locally.
    
    If the dataset argument is not equal to KEYWORD, it tries to load the dataset directly from the disk using load_from_disk().

    If the dataset isn't found locally, it calls parse_dataset() to download and process the dataset from Hugging Face and save 
    it to the local disk.

    If dataset == KEYWORD, it will attempt to load all datasets from the DATA_FOLDER, concatenating them into one large dataset.
    Each dataset in the folder will be read and combined into train and test splits.
    
    """

    if dataset != KEYWORD:

        dataset_folder = os.path.join(DATA_FOLDER, dataset.replace('/', '@'))

        # Check if folder exits
        if os.path.exists(dataset_folder):

            tokenized_dataset = load_from_disk(dataset_folder)
        
        else:
            try:
                tokenized_dataset = parse_dataset(dataset_name=dataset, limit=limit)
            except Exception as e:
                raise Exception('There has been an error while downloading the data from Hugging face:\n' + e, )

    else:
        # Process all datasets in the data folder
        tokenized_dataset = DatasetDict({
            "train": Dataset.from_dict({}),  # Empty Dataset for train
            "test": Dataset.from_dict({})   # Empty Dataset for test
        })

        if os.path.exists(DATA_FOLDER):
     
            for folder in os.listdir(DATA_FOLDER):
                if folder.replace('@', '/') in dataset_names:
                    dataset_names.remove(folder.replace('@', '/'))
                dataset_folder = os.path.join(DATA_FOLDER, folder)
                if os.path.isdir(dataset_folder):
                    try:
                        tokenized_dataset_read = load_from_disk(dataset_folder)

                        if tokenized_dataset:
                            tokenized_dataset['train'] = concatenate_datasets([tokenized_dataset['train'], tokenized_dataset_read['train']])
                            tokenized_dataset['test'] = concatenate_datasets([tokenized_dataset['test'], tokenized_dataset_read['test']])

                    except Exception as e:
                        print(f"Failed to load dataset from {dataset_folder}: {e}")
        
        for dataset in dataset_names:
            print('Parsing: ', dataset)
            tokenized_dataset_read = parse_dataset(dataset, limit=limit)
            if tokenized_dataset:
                tokenized_dataset['train'] = concatenate_datasets([tokenized_dataset['train'], tokenized_dataset_read['train']])
                tokenized_dataset['test'] = concatenate_datasets([tokenized_dataset['test'], tokenized_dataset_read['test']])

    return tokenized_dataset
    

def parse_dataset(dataset_name, limit = None):

    '''
    Prompts will have the following structure:

    # Instruction:

    # Input:

    # Response:
    '''

    def format_lima_conversation(examples):
        # Join the list into a single string if it's a list of sentences
        joined_conversations = []
        i = 0
        for conv in examples['conversations']:
            
            if len(conv) == 2:
                prompt = ""
                prompt += "Instruction: " + conv[0] +"\n\n" + "Context: \n\n" + "Response: " + conv[1]
                joined_conversations.append(prompt)
            
            else:
                print(len(conv))
                i += 1
        print(i)

        return {"prompt": joined_conversations}
    
    def format_dolly_prompts(examples):
        formatted_prompts = []
    
        for instruction, context, response in zip(examples.get('instruction', []), examples.get('context', []), examples.get('response', [])):
            
            prompt = f"Instruction: {instruction}\n\nContext: {context}\n\nResponse: {response}"
            
            formatted_prompts.append(prompt)

        return {"prompt": formatted_prompts}
    
    def format_alpaca(examples):

        formatted_prompts = []

        for instruction, input_text, output in zip(examples.get('instruction', []), examples.get('input', []), examples.get('output', [])):
            
            prompt = f"Instruction: {instruction}\n\nContext: {input_text}\n\nResponse: {output}"
            
            formatted_prompts.append(prompt)

        return {"prompt": formatted_prompts}
    
    def format_oasst(examples):
        formatted_prompts = []

        for query, answer, lang in zip(examples.get('query', []), examples.get('answer', []), examples.get('lang', [])):
            
            prompt = f"Instruction: {query}\n\nContext: \n\nResponse: {answer}"
            
            formatted_prompts.append(prompt)

        return {"prompt": formatted_prompts}
    
    # Load the dataset
    dataset = load_dataset(dataset_name)

    dataset_folder = os.path.join(DATA_FOLDER, dataset_name.replace('/', '@'))

    # Tokenize the dataset

    if "lima" in dataset_name:
        parsed_dataset = dataset.map(format_lima_conversation, batched=False)
        parsed_dataset = parsed_dataset.remove_columns(['conversations','source'], )

    if "dolly" in dataset_name:
        parsed_dataset = dataset.map(format_dolly_prompts, batched=True)
        parsed_dataset = parsed_dataset.remove_columns(['instruction','context', 'response', 'category'], )

    if "alpaca" in dataset_name:
        parsed_dataset = dataset.map(format_alpaca, batched=True)
        parsed_dataset = parsed_dataset.remove_columns(['instruction','input', 'output'], )

    if "oasst" in dataset_name:
        parsed_dataset = dataset.map(format_oasst, batched=True)
        parsed_dataset = parsed_dataset.remove_columns(['lang','message_id', 'parent_id', 'user_id', 'created_date', 'query', 'answer', 'review_count', 'answer_len'], )

    if 'test' not in parsed_dataset.keys():
        # Split the dataset into 80/20 train/test if test split is missing
        train_test_split = parsed_dataset['train'].train_test_split(test_size=0.2)
        parsed_dataset = DatasetDict({
            "train": train_test_split['train'],
            "test": train_test_split['test']
        })

    if limit:
        train_limit = limit
        test_limit = int(limit * 0.2)

        # Shuffle datasets and apply limits
        parsed_dataset['train'] = parsed_dataset['train'].shuffle(seed=42).select(range(min(len(parsed_dataset['train']), train_limit)))
        parsed_dataset['test'] = parsed_dataset['test'].shuffle(seed=42).select(range(min(len(parsed_dataset['test']), test_limit)))

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    # Check if dataset folder exists
    dataset_folder = os.path.join(DATA_FOLDER, dataset_name.replace('/', '@'))
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Save the tokenized dataset to the specified directory
    parsed_dataset.save_to_disk(dataset_folder)

    return parsed_dataset

if __name__ == "__main__":

    # parse_dataset("databricks/databricks-dolly-15k")

    tokenized_dataset = get_dataset(KEYWORD, 15000)

    print('Train: ', len(tokenized_dataset['train']))
    print('Train: ', len(tokenized_dataset['test']))
