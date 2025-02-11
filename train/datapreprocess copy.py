import nltk
from typing import Tuple
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM

from config import DataConfig
from loguru import logger

nltk.download("punkt")
nltk.download("punkt_tab")

def process_data(data: DataConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM) -> Tuple[DatasetDict, DataCollatorForSeq2Seq]:
    """
    Processes a dataset for sequence-to-sequence task.

    This function loads a dataset from Hugging Face, preprocesses the input and target text 
    using a tokenizer, tokenizes the dataset (if not already saved), and prepares a 
    data collator for batch processing during training.

    Args:
        data (DataConfig): An object containing the dataset configuration, including:
            - data.dataset_path (str): The Hugging Face dataset name.
            - data.dataset_subset (str): The specific subset to load.
            - data.input_text_column (str): Column name containing input text.
            - data.label_text_column (str): Column name containing target text (labels).
            - data.max_input_length (int): Maximum token length for inputs.
            - data.max_target_length (int): Maximum token length for labels.
            - data.sample (int, optional): Number of samples to use for training/testing.
            - data.test_sample (int, optional): Number of samples to use for testing.
            - data.load_tokenized_dataset (bool): Whether to load pre-tokenized dataset.
            - data.save_tokenized_dataset (bool): Whether to save tokenized dataset.
        tokenizer (transformers.AutoTokenizer): The tokenizer to preprocess text.
        model (transformers.AutoModelForSeq2SeqLM): The model used to align data collator settings.

    Returns:
        Tuple[DatasetDict, DataCollatorForSeq2Seq]: 
            - The processed and tokenized dataset in a `DatasetDict`.
            - A data collator instance for batched training.

    Steps:
        1. Load dataset from Hugging Face.
        2. Optionally sample a subset of the data for training/testing.
        3. Preprocess text by tokenizing input and labels.
        4. Either load existing tokenized data or process and save a new tokenized dataset.
        5. Initialize a `DataCollatorForSeq2Seq` for batch processing.
        6. Remove unused columns and finalize dataset for training.
    """
    def preprocess_function(examples):
        """
        Tokenizes input and target text while ensuring proper truncation.

        Args:
            examples (dict): A batch of dataset samples.

        Returns:
            dict: Tokenized inputs and labels.
        """
        model_inputs = tokenizer(
            examples[data.input_text_column],
            max_length=data.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            examples[data.label_text_column], max_length=data.max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    #load_cache_dir
    dataset_hf = load_dataset(data.dataset_path, name=data.dataset_subset)
    logger.info(f"Loaded dataset: {data.dataset_path}, name={data.dataset_subset}")

    #TODO Fixed sample for training
    if data.sample:
        #TODO add seed as var
        train_subset = dataset_hf["train"].shuffle(seed=42).select(range(data.sample))
        test_subset = dataset_hf["test"].shuffle(seed=42).select(range(110))

        # Combine the subsets into a new DatasetDict
        dataset_hf = DatasetDict(
            {
                "train": train_subset,
                "test": test_subset,
            }
        )
        logger.info(f'Selected {data.sample} rows for "train" and {data.test_sample} for "test"')
            
            
    if data.load_tokenized_dataset:
        tokenized_datasets = load_from_disk("tokenized_datasets")
        logger.info("Finished loading tokenized dataset.")
    else:
        tokenized_datasets = dataset_hf.map(preprocess_function, batched=True)
        logger.info("Finished preprocessing tokenized dataset.")
        if data.save_tokenized_dataset:
            tokenized_datasets.save_to_disk("tokenized_datasets")
            logger.info("Tokenized dataset saved to dir tokenized_datasets.")

    #TODO To move data collator       
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    #just leave this
    tokenized_datasets = tokenized_datasets.remove_columns(
        dataset_hf["train"].column_names
    )
    features = [tokenized_datasets["train"][i] for i in range(2)]
    data_collator(features)
    logger.info("Data ready for training.")
    
    return tokenized_datasets, data_collator
