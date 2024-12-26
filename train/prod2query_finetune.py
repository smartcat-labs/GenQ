import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
import torch
import csv
from loguru import logger
import os
from datetime import datetime

from config import Configuration
from datapreprocess import process_data
import eval as e

"""
Script for fine-tuning a base model for text-to-query generation.
    This script enables fine-tuning of a pre-trained sequence-to-sequence model to generate queries from text inputs.
    It uses Hugging Face's `Seq2SeqTrainer` for training and evaluation and logs progress and results.
    Features:
    - Reads configuration from a YAML file for flexible setup.
    - Processes input data for training and evaluation using tokenization and data collation.
    - Fine-tunes a model with support for evaluation and logging.
    - Tracks training and evaluation metrics, saving them to CSV files for further analysis.
    Requirements:
    - Install dependencies: transformers, torch, loguru, csv, PyYAML, and additional custom modules (config, datapreprocess, eval).
    How to run:
    1. Prepare a configuration file (YAML format) specifying:
       - Model checkpoint, training arguments, data paths, and evaluation settings.
    2. Execute the script from the terminal:
       python prod2query_finetune.py -c config.yaml -r 'finetuned-amazon-product2query' --log_level INFO
    Example configuration file (config.yaml):
    --------------------------------------------------
    data:
      train_data: "path/to/train_data.jsonl"
      test_data: "path/to/test_data.jsonl"
    train:
      model_checkpoint: "t5-base"
      output_dir_name: "finetuned-model-output"
      evaluation_strategy: "epoch"
      learning_rate: 5e-5
      batch_size: 16
      num_train_epochs: 3
      save_total_limit: 2
      predict_with_generate: true
      logging_strategy: "steps"
      save_strategy: "epoch"
      push_to_hub: false
      load_best_model_at_end: true
      metric_for_best_model: "eval_loss"
      greater_is_better: false
      report_to: none
    --------------------------------------------------
    Outputs:
    - Fine-tuned model saved in the specified directory.
    - Metrics and evaluation results saved in:
        - `evaluation_metrics.csv`: Training and evaluation metrics per step.
        - `final_metrics.csv`: Final metrics after training completion.
    - Logs stored in `finetuning.log`.
    Logs:
    - Training progress and configuration details are logged for traceability.

"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Finetune model with arguments")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "-r", "--output_path", type=str, default="finetuned-amazon-product2query"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    return parser.parse_args()


def run_training(args):
    config = Configuration.from_yaml(args.config)

    config.to_yaml("config.yaml")  # Save the configuration

    logger.add("finetuning.log", level=args.log_level)
    logger.info("=======Starting=======")
    logger.info("Configuration saved to 'config.yaml'")

    data = config.data
    train = config.train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device to be used is {device}")

    model = AutoModelForSeq2SeqLM.from_pretrained(train.model_checkpoint)
    model.to(device=device)

    tokenizer = AutoTokenizer.from_pretrained(train.model_checkpoint)

    logger.info(f"Model and Tokenizer: {train.model_checkpoint} is ready.")

    tokenized_datasets, data_collator = process_data(data, tokenizer, model)

    # Show the training loss with every epoch
    logging_steps = len(tokenized_datasets["train"]) // train.batch_size
    logger.info(f'Logging steps is "{logging_steps}"')

    args = Seq2SeqTrainingArguments(
        output_dir=f"{train.output_dir_name}",
        eval_strategy=train.evaluation_strategy,
        learning_rate=train.learning_rate,
        per_device_train_batch_size=train.batch_size,
        per_device_eval_batch_size=train.batch_size,
        weight_decay=train.weight_decay,
        save_total_limit=train.save_total_limit,
        num_train_epochs=train.num_train_epochs,
        predict_with_generate=train.predict_with_generate,
        logging_steps=logging_steps,
        logging_strategy=train.logging_startegy,
        save_strategy=train.save_strategy,
        push_to_hub=train.push_to_hub,
        load_best_model_at_end=train.load_best_model_at_end,
        metric_for_best_model=train.metric_for_best_model,
        greater_is_better=train.greater_is_better,
        report_to=train.report_to,
    )
    logger.info("Training arguments prepared.")

    class PrinterCallback(TrainerCallback):
        def __init__(self):
            self.metrics_file = "evaluation_metrics.csv"
            self.final_metrics_file = "final_metrics.csv"

            # Dictionary to store partial metrics keyed by (epoch, step)
            self.metrics_by_step = {}

            # Ensure the evaluation metrics CSV file exists and has headers
            if not os.path.exists(self.metrics_file):
                with open(self.metrics_file, "w", newline="") as f:
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
            if not os.path.exists(self.final_metrics_file):
                with open(self.final_metrics_file, "w", newline="") as f:
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
            have_evaluation = (
                "eval_loss" in combined_logs or "eval_rouge1" in combined_logs
            )

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

    printer_callback = PrinterCallback()

    def compute_metrics_wrapper(eval_pred):
        return e.compute_metrics(eval_pred, tokenizer)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_wrapper,
        callbacks=[printer_callback],
    )
    logger.info("Trainer ready.")
    logger.info("(^o^) Training started (^o^)")

    trainer.train()
    logger.info("^ω^ Finished training ^ω^")


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
