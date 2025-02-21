import os
import csv
import argparse
import evaluate
import nltk
import yaml
import torch
from datetime import datetime
from typing import List, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from pathlib import Path
from loguru import logger

"""
Script for evaluating query generation models and computing ROUGE scores.
    This script processes a dataset of text descriptions and target queries, evaluates multiple
    pre-trained sequence-to-sequence models by generating queries, and calculates ROUGE scores
    to assess similarity between generated and target queries. The results are saved in a CSV file.
    Features:
    - Reads configuration from a YAML file.
    - Loads and processes datasets from local files or Hugging Face's datasets library.
    - Supports multiple model checkpoints for evaluation.
    - Computes ROUGE metrics for each generated query.
    - Saves evaluation results, including ROUGE scores, to a CSV file.

    Requirements:
    - Install dependencies: transformers, datasets, nltk, evaluate, torch, loguru, tqdm, pyyaml, csv.

    How to run:
    1. Prepare a configuration file (YAML format) specifying:
       - `input_text_column`: Column name for input descriptions (default: "description").
       - `label_text_column`: Column name for target queries (default: "short_query").
       - `dataset`: Path to the dataset or Hugging Face dataset name.
       - `model_paths`: List of model checkpoint paths for evaluation.
       - `sample`: (Optional) Number of examples to sample from the dataset for testing.
       - 'seed': (Optional) If sample is being used set seed to reproduce the sample.
    2. Run the script from the terminal:
       python eval/checkpoint_eval.py config/eval_config.yaml

    Example configuration file (eval_config.yaml):
    --------------------------------------------------
    input_text_column: "description"
    label_text_column: "short_query"
    dataset: "../metadata_with_queries.jsonl"
    model_paths:
      - "path/to/checkpoint1"
      - "path/to/checkpoint2"
    sample: 500
    seed: 42
    --------------------------------------------------
    
    Outputs:
    - CSV file ("generated_results.csv") containing:
        - Input titles and descriptions and target queries.
        - Generated queries for each model.
        - ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores.
    - Log file ("evaluation.log") with detailed execution steps and errors.
    Logs:
    - Execution details and errors are logged for debugging purposes.
    Notes:
    - Ensure that NLTK punkt and punkt_tab tokenizer is downloaded as part of the script's setup.
    - Model paths should point to valid Hugging Face model checkpoints.
"""

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the script with a config file.")
    parser.add_argument("config_file", type=str, help="Path to the YAML config file")
    return parser.parse_args()

def load_config(config_file: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_file(str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters
    """
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config=yaml.safe_load(f)
        logger.info(f"Configuration loaded from '{config_file}'.")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_file}' not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise

def get_device() -> torch.device:
    """
    Get the available device (GPU if available, otherwise CPU).

    Returns:
        torch.device: The device to be used.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device

def prepare_input_text(example: dict, input_columns: List[str]) -> str:
    """
    Combine multiple input columns into a single text string.

    Args:
        example (dict): A dataset example.
        input_columns (List[str]): List of column names to combine.

    Returns:
        str: Combined input text.
    """
    texts = [str(example[col]) for col in input_columns if col in example and example[col] is not None]
    return "\n\n".join(texts)

def generate_output_and_compute_rouge(
    model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, input_text: str, target_query: str, rouge_score: evaluate.EvaluationModule
) -> Tuple[str, dict]:
    """
    Generate output text from from the model and compute ROUGE scores.

    Args: 
        mode: The seq2seq model.
        tokenizer: The associated tokenizer.
        input_text (str): The input text (combined "title" + "description").
        target_query (str): The target query text
        rouge_metric: The ROUGE metric object

    Returns:
        tuple: Generated text and dictionary of ROUGE scores.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    generated_ids = model.generate(
        inputs["input_ids"], max_length=30, num_beams=4, early_stopping=True
    )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Compute ROUGE scores
    decoded_target = target_query.strip()
    decoded_generated = generated_text.strip()

    # ROUGE expects a newline after each sentence
    decoded_generated = "\n".join(sent_tokenize(decoded_generated))
    decoded_target = "\n".join(sent_tokenize(decoded_target))

    scores = rouge_score.compute(
        predictions=[decoded_generated], references=[decoded_target]
    )
    return generated_text, scores


def run_evaluation(config: dict) -> None:
    """
    Run the evaluation pipeline.

    Args:
        config(dict): Configuration parameters from the YAML file.
    
    Returns:
        None
    """
    device = get_device()

    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(f"eval/runs/{dt}")
    output_path.mkdir(parents=True, exist_ok=True)

    log_level = config.get("log_level", "INFO")
    logger.add(f"{output_path}/evaluation.log", level=log_level)

    with open(f"{output_path}/eval_config.yaml", 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    logger.info(f"Configuration saved to '{output_path}/eval_config.yaml'")

    logger.info("=======Starting Evaluation=======")

    # Download necessary NLTK data
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    #Retrieve configuration parameters.
    input_text_column = config.get("input_text_column", ["title", "description"])
    label_text_column = config.get("label_text_column", "short_query")
    dataset = config.get("dataset")
    name = config.get("name")
    split = config.get("split", "test")
    model_paths = config.get("model_paths", [])
    #For a sample of 500 examples the evaluation will last 15-20 minutes using CPU
    sample = config.get("sample") #500)
    seed = config.get("seed", 42)
    # Load the dataset from Hugging Face
    dataset_test = load_dataset(
        dataset,
        name=name,
        split=split,
    )

    logger.info(f"Loaded dataset {dataset}")

    if sample:
        dataset_test = dataset_test.shuffle(seed=seed).select(range(sample))
        logger.info(f"Dataset loaded with {len(dataset_test)} examples")

    # Load the models and tokenizers
    models = [
        AutoModelForSeq2SeqLM.from_pretrained(path).to(device) for path in model_paths
    ]
    tokenizers = [AutoTokenizer.from_pretrained(path) for path in model_paths]
    logger.info("Loaded all models")

    # ROUGE scorer setup
    rouge_score = evaluate.load("rouge")
    logger.info("Loaded metric")

    # Prepare CSV file
    output_file = (f"{output_path}/generated_results.csv")    

    header = [
        "title",
        "description",
        "input_text",
        "target_query",
        "model",
        "generated_output",
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeLsum",
    ]

    # Iterate over the test data with tqdm progress bar
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    logger.info(f"Written header in {output_file} csv file")

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for i, example in tqdm(enumerate(dataset_test), total=len(dataset_test), desc="Evaluating"):
            input_text = prepare_input_text(example, input_text_column)
            title = str(example.get("title", ""))
            description = str(example.get("description", ""))
            target_query = str(example.get(label_text_column, ""))

            # For each model checkpoint, generate a row in the CSV
            for j, (model, tokenizer) in enumerate(zip(models, tokenizers)):
                checkpoint_path = model_paths[j]
                model_name = os.path.basename(checkpoint_path)

                # Simple check to verify name is part of the path
                if model_name not in checkpoint_path:
                    logger.warning(
                        f"Checkpoint name '{model_name}' does not fully match path '{checkpoint_path}'"
                    )

                # Generate output and compute ROUGE
                generated_output, rouge_scores = generate_output_and_compute_rouge(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=input_text,
                    target_query=target_query,
                    rouge_score=rouge_score,
                )

                # Round scores to two decimals and multiply by 100
                # (They are floats in the range [0,1], so *100 is typical for ROUGE in %)
                r1 = round(rouge_scores["rouge1"] * 100, 2)
                r2 = round(rouge_scores["rouge2"] * 100, 2)
                rl = round(rouge_scores["rougeL"] * 100, 2)
                rlsum = round(rouge_scores["rougeLsum"] * 100, 2)

                row = [
                    title,
                    description,
                    input_text,
                    target_query,
                    model_name,
                    generated_output,
                    r1,
                    r2,
                    rl,
                    rlsum,
                ]
                writer.writerow(row)

    logger.info(f"Results have been written to '{output_file}'")
    logger.info("=======Evaluation Completed======")

def main():
    args = parse_args()
    config = load_config(args.config_file)
    run_evaluation(config)


if __name__ == "__main__":
    main()
