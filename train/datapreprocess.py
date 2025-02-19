import nltk
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer

from config import DataConfig
from loguru import logger

nltk.download("punkt")
nltk.download("punkt_tab")

def process_data(data: DataConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    """
    Processes a dataset for sequence-to-sequence task.

    This function loads a dataset from Hugging Face, preprocesses the input and target text 
    using a tokenizer, tokenizes the dataset.

    Args:
        data (DataConfig): An object containing the dataset configuration, including:
            - data.dataset_path (str): The Hugging Face dataset name.
            - data.dataset_subset (str): The specific subset to load.
            - data.input_text_column (List[str]): List of column names containing input text.
            - data.label_text_column (str): Column name containing target text (labels).
            - data.max_input_length (int): Maximum token length for inputs.
            - data.max_target_length (int): Maximum token length for labels.
            - data.cache_dir (Path): Path to the cache directory 
            - data.dev (bool): Whether to run the training on a small percentage of the dataset
        tokenizer (transformers.AutoTokenizer): The tokenizer to preprocess text.

    Returns:
        DatasetDict
            - The processed and tokenized dataset in a `DatasetDict`.

    Steps:
        1. Load dataset from Hugging Face.
        2. Optionally process the sample a subset of the data for training/testing.
        3. Preprocess text by tokenizing input and labels.
        4. Process tokenized dataset.
        5. Remove unused columns and finalize dataset for training.
    """

    def extend_dataset(dataset: DatasetDict) -> DatasetDict:
        """
        Extends a dataset by duplicating entries while clearing the 'description' field.

        This function takes an existing `DatasetDict` containing 'train' and 'test' splits, 
        converts them into Pandas DataFrames, and duplicates the entries. In the duplicated 
        rows, the 'description' field is set to an empty string. The extended dataset is then 
        converted back into a `DatasetDict` format.

        Args:
            dataset (DatasetDict): The original dataset containing 'train' and 'test' splits.

        Returns:
            DatasetDict: A new dataset dictionary where both 'train' and 'test' splits have 
                        doubled the number of rows, with half of them having an empty 
                        'description' field.

        Process:
            1. Convert the 'train' and 'test' datasets from `DatasetDict` into Pandas DataFrames.
            2. Create a duplicate of each dataset.
            3. Set the 'description' column to an empty string in the duplicated data.
            4. Concatenate the original and duplicated DataFrames.
            5. Convert the extended DataFrames back into the Hugging Face `Dataset` format.
            6. Return the extended dataset as a `DatasetDict`.
        """

        train = pd.DataFrame(dataset["train"])
        test = pd.DataFrame(dataset["test"])

        train_duplicate = train.copy()
        test_duplicate = test.copy()
        
        difference = [item for item in data.input_text_column if item not in data.extend_columns]

        for column in difference:
            train_duplicate[column] = ""
            test_duplicate[column] = ""

        train_extended = pd.concat([train, train_duplicate], ignore_index=True)
        test_extended = pd.concat([test, test_duplicate], ignore_index=True)

        train_dataset = Dataset.from_pandas(train_extended)
        test_dataset = Dataset.from_pandas(test_extended)

        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        }
        )

        logger.info("Extended the dataset")
        
        return dataset_dict

    def preprocess_function(examples):
        """
        Tokenizes input and target text while ensuring proper truncation.

        Args:
            examples (dict): A batch of dataset samples.

        Returns:
            dict: Tokenized inputs and labels.
        """
        input_texts = [
        "\n\n".join(
            str(examples[col][i]) if col in examples and examples[col][i] is not None else ""
            for col in data.input_text_column
        ).strip()
        for i in range(len(examples[data.input_text_column[0]]))  # Iterate over batch elements
    ]
        model_inputs = tokenizer(
            input_texts,
            max_length=data.max_input_length,
            truncation=True,
            padding = "max_length"
        )
        labels = tokenizer(
            examples[data.label_text_column], max_length=data.max_target_length, truncation=True, padding = "max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    
    logger.info(f"Loaded dataset: {data.dataset_path}")

    if data.dev:
        split_percentage = "[:1%]"
    else:
        split_percentage = ""   

    train_subset = load_dataset(data.dataset_path, name=data.dataset_subset, cache_dir=data.cache_dir, split="train" + split_percentage)
    test_subset = load_dataset(data.dataset_path, name=data.dataset_subset, cache_dir=data.cache_dir, split="test" + split_percentage) 
          
    dataset_hf = DatasetDict(
        {
            "train": train_subset,
            "test": test_subset,
        }
    )
    
    if data.extend_columns:
        dataset_hf = extend_dataset(dataset_hf)
        
    logger.info(f'Selected {len(train_subset)} rows for "train" and {len(test_subset)} for "test"') 

    tokenized_datasets = dataset_hf.map(preprocess_function, batched=True, remove_columns=dataset_hf["train"].column_names)
    logger.info("Finished preprocessing tokenized dataset.")
    
    logger.info("Data ready for training.")
    
    return tokenized_datasets
