import os
import csv
import argparse
import evaluate
import nltk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import yaml
import torch
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
    2. Run the script from the terminal:
       python checkpoint_eval.py eval_config.yaml
    Example configuration file (eval_config.yaml):
    --------------------------------------------------
    input_text_column: "description"
    label_text_column: "short_query"
    dataset: "../metadata_with_queries.jsonl"
    model_paths:
      - "path/to/checkpoint1"
      - "path/to/checkpoint2"
    sample: 500
    --------------------------------------------------
    Outputs:
    - CSV file ("generated_results.csv") containing:
        - Input descriptions and target queries.
        - Generated queries for each model.
        - ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores.
    - Log file ("evaluation.log") with detailed execution steps and errors.
    Logs:
    - Execution details and errors are logged for debugging purposes.
    Notes:
    - Ensure that NLTK punkt and punkt_tab tokenizer is downloaded as part of the script's setup.
    - Model paths should point to valid Hugging Face model checkpoints.
"""


def generate_output_and_compute_rouge(
    model, tokenizer, description, target_query, rouge_score
):
    inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
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


def run_evaluation(
    input_text_column, label_text_column, dataset, name, split, model_paths, sample
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device to be used is {device}")

    # Download necessary NLTK data
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    # Load the dataset from Hugging Face
    dataset_test = load_dataset(
        dataset,
        name=name,
        split=split,
    )

    logger.info(f"Loaded dataset {dataset}")

    if sample:
        dataset_test = dataset_test.shuffle(seed=42).select(range(sample))

    # Load the models and tokenizers
    models = [
        AutoModelForSeq2SeqLM.from_pretrained(path).to(device) for path in model_paths
    ]
    tokenizers = [AutoTokenizer.from_pretrained(path) for path in model_paths]
    logger.info("loaded all models")

    # ROUGE scorer setup
    rouge_score = evaluate.load("rouge")
    logger.info("loaded metric")

    # Prepare CSV file
    output_file = "generated_results.csv"

    # Write header only once
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "description",
            "target_query",
            "checkpoint_name",
            "generated_output",
            "rouge1",
            "rouge2",
            "rougeL",
            "rougeLsum",
        ]
        writer.writerow(header)
    logger.info("written header")

    # Iterate over the test data with tqdm progress bar
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        logger.info("opened csv file")

        for i, example in tqdm(enumerate(dataset_test), total=len(dataset_test)):
            description = example[input_text_column]
            target_query = example[label_text_column]

            # For each model checkpoint, generate a row in the CSV
            for j, (model, tokenizer) in enumerate(zip(models, tokenizers)):
                checkpoint_path = model_paths[j]
                checkpoint_name = os.path.basename(checkpoint_path)

                # Simple check to verify name is part of the path
                if checkpoint_name not in checkpoint_path:
                    logger.warning(
                        f"Checkpoint name '{checkpoint_name}' does not fully match path '{checkpoint_path}'"
                    )

                # Generate output and compute ROUGE
                generated_output, rouge_scores = generate_output_and_compute_rouge(
                    model=model,
                    tokenizer=tokenizer,
                    description=description,
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
                    description,
                    target_query,
                    checkpoint_name,
                    generated_output,
                    r1,
                    r2,
                    rl,
                    rlsum,
                ]
                writer.writerow(row)

    print(f"Results have been written to '{output_file}'")


if __name__ == "__main__":
    logger.add("evaluation.log", level="INFO")
    logger.info("=======Starting=======")

    parser = argparse.ArgumentParser(description="Run the script with a config file.")
    parser.add_argument("config_file", help="Path to the YAML config file")
    args = parser.parse_args()

    try:
        with open(args.config_file) as f:
            config = yaml.safe_load(f)

        input_text_column = config.get("input_text_column", "description")
        label_text_column = config.get("label_text_column", "short_query")
        dataset = config.get("dataset", "smartcat/Amazon_Sample_Metadata_2023")
        name = config.get("name", None)
        split = config.get("split", None)
        model_paths = config.get("model_paths", [])
        sample = config.get("sample", 500)

        logger.info("Successfully read config file.")

        run_evaluation(
            input_text_column,
            label_text_column,
            dataset,
            name,
            split,
            model_paths,
            sample,
        )

    except FileNotFoundError:
        logger.warning("Config file not found.")
    except yaml.YAMLError as e:
        logger.error(f"Error reading the config file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
