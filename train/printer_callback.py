import csv
import glob
from datetime import datetime
from transformers import TrainerCallback

class PrinterCallback(TrainerCallback):
    """
    A custom TrainerCallback that logs training and evaluation metrics to CSV files.
    """
    def __init__(self):
        self.metrics_file = "evaluation_metrics.csv"
        self.final_metrics_file = "final_metrics.csv"

        # Dictionary to store partial metrics keyed by (epoch, step)
        self.metrics_by_step = {}

        # Ensure the evaluation metrics CSV file exists and has headers
        if not glob.glob(self.metrics_file):
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
        if not glob.glob(self.final_metrics_file):
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
