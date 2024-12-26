import nltk
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import DataCollatorForSeq2Seq

from loguru import logger

nltk.download("punkt")
nltk.download("punkt_tab")

def process_data(data, tokenizer, model):
    
    def preprocess_function(examples):
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
    
    
    dataset_hf = load_dataset(data.dataset_path, name=data.dataset_subset)
    logger.info(f"Loaded dataset: {data.dataset_path}, name={data.dataset_subset}")

    if data.sample:
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
            
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    tokenized_datasets = tokenized_datasets.remove_columns(
        dataset_hf["train"].column_names
    )
    features = [tokenized_datasets["train"][i] for i in range(2)]
    data_collator(features)
    logger.info("Data ready for training.")
    
    return tokenized_datasets, data_collator
