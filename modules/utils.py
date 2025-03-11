import csv
import torch
import yaml
from loguru import logger
from datetime import datetime
from transformers import TrainerCallback, AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path
from evaluate import load
from typing import List


def get_device() -> torch.device:
    """
    Returns the best available device (CUDA, MPS, or CPU).

    Returns:
        torch.device: The selected device for model training.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    return device


def load_config(config_file: str, type: str) -> dict:
    """
    Load the configuration from a YAML file and set default values for missing keys.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file '{config_file}' not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        raise
    if type == "eval":
        defaults = {
            "input_text_columns": ["title", "description"],
            "label_text_column": "short_query",
            "dataset": "smartcat/Amazon-2023-GenQ",
            "name": None,
            "split": "test",
            "batch_size": 16,
            "model_paths": ["BeIR/query-gen-msmarco-t5-base-v1"],
            "sample": None,
            "seed": 42,
            "log_level": "INFO",
        }
    elif type == "analysis":
        defaults = {
            "results_path": "/home/petar/Documents/trainings/14-02/generated_results.csv",
            "save_zeros": False,
            "save_outperformed": False,
            "save_best": False,
            "save_worst": False,
            "compute_similarity": False,
            "compare_models": [0, 1],
        }

    for key, value in defaults.items():
        if key not in config:
            config[key] = value
            logger.warning(f"Key '{key}' not found in config. Using default: {value}")

    logger.info(f"Loaded config from {config_file}")
    return config


class PrinterCallback(TrainerCallback):
    """
    A custom TrainerCallback that logs training and evaluation metrics to CSV files.
    """

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.metrics_file = self.save_dir / "evaluation_metrics.csv"
        self.final_metrics_file = self.save_dir / "final_metrics.csv"

        # Dictionary to store partial metrics keyed by (epoch, step)
        self.metrics_by_step = {}

        # Ensure the evaluation metrics CSV file exists and has headers
        if not self.metrics_file.exists():
            with self.metrics_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "epoch",
                        "step",
                        "loss",
                        "grad_norm",
                        "learning_rate",
                        "eval_loss",
                        "eval_rouge1",
                        "eval_rouge2",
                        "eval_rougeL",
                        "eval_rougeLsum",
                        "eval_runtime",
                        "eval_samples_per_second",
                        "eval_steps_per_second",
                    ]
                )

        # Ensure the final metrics CSV file exists and has headers
        if not self.metrics_file.exists():
            with self.metrics_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "train_runtime",
                        "train_samples_per_second",
                        "train_steps_per_second",
                        "train_loss",
                        "epoch",
                        "step",
                    ]
                )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            logs = {}

        # Remove "total_flos" if it exists
        _ = logs.pop("total_flos", None)

        # Add epoch and step info
        logs["epoch"] = state.epoch
        logs["step"] = state.global_step

        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # If this is final metrics (has train_runtime), write directly to final_metrics.csv
        if "train_runtime" in logs:
            with open(self.final_metrics_file, "a", newline="") as f:
                writer = csv.writer(f)
                row = [
                    timestamp,
                    logs.get("train_runtime", ""),
                    logs.get("train_samples_per_second", ""),
                    logs.get("train_steps_per_second", ""),
                    logs.get("train_loss", ""),
                    logs.get("epoch", ""),
                    logs.get("step", ""),
                ]
                writer.writerow(row)
            return

        # For training/evaluation logs, combine with any existing metrics for the same (epoch, step)
        epoch_step_key = (logs.get("epoch"), logs.get("step"))
        if epoch_step_key not in self.metrics_by_step:
            self.metrics_by_step[epoch_step_key] = {}

        # Merge new logs into existing entry
        self.metrics_by_step[epoch_step_key].update(logs)

        # Check if we have both training and evaluation metrics
        # We'll consider we have both sets if we have a "loss" (training) and "eval_loss" (evaluation) key
        combined_logs = self.metrics_by_step[epoch_step_key]
        have_training = (
            "loss" in combined_logs
            or "grad_norm" in combined_logs
            or "learning_rate" in combined_logs
        )
        have_evaluation = "eval_loss" in combined_logs or "eval_rouge1" in combined_logs

        # If we have both sets of metrics, write them to the CSV
        if have_training and have_evaluation:
            with open(self.metrics_file, "a", newline="") as f:
                writer = csv.writer(f)
                row = [
                    timestamp,
                    combined_logs.get("epoch", ""),
                    combined_logs.get("step", ""),
                    combined_logs.get("loss", ""),
                    combined_logs.get("grad_norm", ""),
                    combined_logs.get("learning_rate", ""),
                    combined_logs.get("eval_loss", ""),
                    combined_logs.get("eval_rouge1", ""),
                    combined_logs.get("eval_rouge2", ""),
                    combined_logs.get("eval_rougeL", ""),
                    combined_logs.get("eval_rougeLsum", ""),
                    combined_logs.get("eval_runtime", ""),
                    combined_logs.get("eval_samples_per_second", ""),
                    combined_logs.get("eval_steps_per_second", ""),
                ]
                writer.writerow(row)

            # Remove this entry from memory since we've written it out
            del self.metrics_by_step[epoch_step_key]


class RougeScorer:
    """
    Class for calculating ROUGE scores in batches.
    """

    def __init__(self):
        self.scorer = load("rouge")

    def compute_batch(self, preds: List[str], refs: List[str]) -> List[dict]:
        """
        Compute ROUGE scores for a batch of predictions and references.

        Args:
            preds (List[str]): List of predicted texts.
            refs (List[str]): List of reference texts.

        Returns:
            results List[dict]: List of dictionaries containing ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores for each prediction.
        """
        try:
            scores = self.scorer.compute(
                predictions=preds, references=refs, use_aggregator=False
            )
            return [
                {
                    "rouge1": round(scores["rouge1"][i] * 100, 2),
                    "rouge2": round(scores["rouge2"][i] * 100, 2),
                    "rougeL": round(scores["rougeL"][i] * 100, 2),
                    "rougeLsum": round(scores["rougeLsum"][i] * 100, 2),
                }
                for i in range(len(preds))
            ]
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error in compute_batch: {e}")
            return [
                {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
            ] * len(preds)


class Model:
    """
    Class for loading models and generating texts in batches.
    """

    def __init__(self, model_path: str, device: torch.device, cache_dir: str = None):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else None,
            cache_dir=cache_dir,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.device = device
        self.model.eval()

    def generate(self, input_texts: List[str]) -> List[str]:
        """
        Generate text from the input list of strings using the model.

        Args:
            input_texts (List[str]): List of input strings to generate text from.

        Returns:
            List[str]: List of generated texts.
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            generated_ids = self.model.generate(
                inputs.input_ids, max_length=30, num_beams=4, early_stopping=True
            )
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
