import nltk
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

from modules.train.config import DataConfig
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

    def preprocess_function(examples):
        """
        Tokenizes input and target text while ensuring proper truncation.

        Args:
            examples (dict): A batch of dataset samples.

        Returns:
            dict: Tokenized inputs and labels.
        """
        input_texts = [
            "\n\n".join(filter(None, (sample))).strip()
            for sample in zip(*[examples[col] for col in data.input_text_column])
        ]

        model_inputs = tokenizer(
            input_texts, max_length=data.max_input_length, truncation=True
        )
        labels = tokenizer(
            examples[data.label_text_column],
            max_length=data.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    logger.info(f"Loaded dataset: {data.dataset_path}")

    if data.dev:
        split_samples = "[:100]"
    else:
        split_samples = ""

    train_subset, test_subset = load_dataset(
        data.dataset_path,
        name=data.dataset_subset,
        cache_dir=data.cache_dir,
        split=["train" + split_samples, "test" + split_samples],
    )

    dataset_hf = DatasetDict({"train": train_subset, "test": test_subset})
    logger.info(
        f'Selected {len(train_subset)} rows for "train" and {len(test_subset)} for "test"'
    )

    tokenized_datasets = dataset_hf.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_hf["train"].column_names,
    )
    logger.info("Finished preprocessing tokenized dataset.")

    logger.info("Data ready for training.")

    return tokenized_datasets
