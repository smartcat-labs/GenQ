import os
import csv
import argparse
import yaml
import nltk
from typing import List
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset
from modules.utils import get_device, load_eval_config, RougeScorer, Model
from torch.utils.data import DataLoader
from nltk.tokenize import sent_tokenize

"""
Script for Evaluating Text-to-Query Generation Models
    This script evaluates pre-trained sequence-to-sequence models for generating queries from text inputs. 
    It uses Hugging Face's transformers library for model inference and computes ROUGE scores to measure the quality of generated queries. 
    The script is designed to be flexible, allowing evaluation of multiple models on a dataset with customizable configurations.

Features:
    Flexible Configuration: Reads evaluation settings from a YAML file, including dataset details, model paths, and evaluation parameters.
    Dynamic Input Handling: Processes input data dynamically based on the configuration, allowing for different input columns and dataset formats.
    Batch Processing: Evaluates models in batches for efficient inference and scoring.
    ROUGE Scoring: Computes ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores to evaluate the quality of generated queries.
    Results Logging: Saves evaluation results, including input text, target text, generated queries, and ROUGE scores, to a CSV file for further analysis.

Requirements:
    Install dependencies: transformers, torch, datasets, loguru, nltk, PyYAML, tqdm.

How to Run:
    Prepare a configuration file (YAML format) specifying:
        Dataset details (name, split, input columns, and target column).
        Model paths (local or Hugging Face Hub).
        Evaluation settings (batch size, sample size, etc.).
        Execute the script from the terminal:
    Run with:
        python -m modules.eval.model_eval -c config/eval_config.yaml

Example Configuration File (eval_config.yaml):
    dataset: smartcat/Amazon-2023-GenQ
    name: null
    split: test
    input_text_columns: ["title", "description"]
    label_text_column: short_query
    model_paths:
    - smartcat/T5-product2query-finetune-v1
    - BeIR/query-gen-msmarco-t5-base-v1
    batch_size: 16
    sample: 500
    seed: 42
    log_level: INFO

Outputs:
    Evaluation Results: Saved in a CSV file (results.csv) with the following columns:
        Input columns (e.g., title, description).
        Combined input text.
        Target text.
        Model name.
        Generated query.
        ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores.

    Logs: Stored in the console and in a log file for traceability.
    Configuration: Used in the script.


Script Workflow
    1. Load Configuration:
        The script reads the configuration file (eval_config.yaml) to set up evaluation parameters, 
        including dataset details, model paths, and batch size.

    2. Initialize Components:
        Sets up the device (GPU or CPU).
        Initializes the ROUGE scorer and loads the specified models.

    3. Load and Prepare Dataset:
        Loads the dataset using Hugging Face's datasets library.
        Optionally samples a subset of the dataset for faster evaluation.
        Preprocesses the target text by tokenizing sentences and joining them with newlines (required for ROUGE-Lsum).

    4. Batch Processing:
        Uses a DataLoader to process the dataset in batches.
        Combines input columns dynamically based on the configuration.

    5. Model Inference:
        Generates queries for each batch of input texts using the specified models.
        Computes ROUGE scores for the generated queries compared to the target text.

    6. Save Results:
        Appends evaluation results (input text, target text, generated queries, and ROUGE scores) to a CSV file.

"""

# Download necessary NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

def run_evaluation(config: dict) -> None:
    """
    Run the evaluation pipeline using the provided configuration.

    Args:
        config (dict): The configuration dictionary.
    """
    
    device = get_device()
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(f"modules/eval/runs/{dt}")
    output_path.mkdir(parents=True, exist_ok=True)

    logger.add(f"{output_path}/eval.log", level=config["log_level"])
    logger.info("=======Starting=======")
    with open(f"{output_path}/eval_config.yaml", 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    logger.info(f"Configuration saved to '{output_path}/eval_config.yaml'")
    
    # Initialize components
    rouge_scorer = RougeScorer()
    models = [Model(path, device) for path in config['model_paths']]
    logger.info(f"Loaded models: {config['model_paths']}")


    # Load and prepare dataset
    dataset = load_dataset(config['dataset'], name=config['name'], split=config['split'])
    if config.get('sample'):
        dataset = dataset.shuffle(config['seed']).select(range(config['sample']))
        
    logger.info(f"Loaded dataset {config['dataset']}, with {len(dataset)} examples.")

    # Preprocess targets once
    dataset = dataset.map(lambda x: {'preprocessed_target': '\n'.join(sent_tokenize(str(x[config['label_text_column']])))},
                          load_from_cache_file=False)

    # Prepare dataloader
    def collate_fn(batch: List[dict]) -> dict:
        """
        Collate function for the DataLoader to process batches.

        Args:
            batch (List[dict]): A list of dataset examples.

        Returns:
            dict: A dictionary containing processed batch data.
        """
        return {
            'input_texts': ['\n\n'.join([str(x[col]) for col in config['input_text_columns']]) for x in batch],
            'targets': [x['preprocessed_target'] for x in batch],
            'metadata': [
                tuple(str(x.get(col, "")) for col in config["input_text_columns"]) for x in batch],
        }

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_fn)
    logger.info(f"Prepared batches of size {config['batch_size']}.")

    # Prepare CSV
    csv_path = output_path / "generated_results.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            *config["input_text_columns"],
            "input_text",
            "target",
            "model",
            "generated_output",
            "rouge1",
            "rouge2",
            "rougeL",
            "rougeLsum",
        ])
    logger.info("Initialized csv.")

    # Evaluation loop
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_texts = batch['input_texts']
        targets = batch['targets']
        metadata = batch['metadata']

        rows = []
        for model in models:
            generated = model.generate(input_texts)
            processed_gen = ['\n'.join(sent_tokenize(g.strip())) for g in generated]
            scores = rouge_scorer.compute_batch(processed_gen, targets)
            
            for meta, inp, tgt, gen, score in zip(metadata, input_texts, targets, generated, scores):
                rows.append([
                    *meta,
                    inp,
                    tgt,
                    os.path.basename(model.tokenizer.name_or_path), #This is simpler for handling both local and HF models
                    gen,
                    score["rouge1"],
                    score["rouge2"],
                    score["rougeL"],
                    score["rougeLsum"],
                ])

        # Batch write to CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)

    logger.info(f"Evaluation complete. Results saved to {csv_path}")

def main():

    parser = argparse.ArgumentParser(description="Run evaluation with a config file.")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    config = load_eval_config(args.config)

    run_evaluation(config)

if __name__ == "__main__":
    main()