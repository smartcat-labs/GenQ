import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from pathlib import Path
from loguru import logger
from datetime import datetime
from modules.train.config import Configuration
from modules.train.datapreprocess import process_data
from modules.train.eval import compute_metrics
from modules.utils import PrinterCallback, get_device
from transformers import set_seed

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
    - Install dependencies: transformers, torch, loguru, csv, PyYAML.
    How to run:
    1. Prepare a configuration file (YAML format) specifying:
       - Model checkpoint, training arguments, data paths, and evaluation settings.
    2. Execute the script from the terminal:
        python -m modules.train.train -c config/config.yaml -o 'finetuned-amazon-product2query' --log_level INFO 
    (Optional) to run on a smaller percentage of the dataset execute the script from the terminal with:
        python -m modules.train.train -c config/test_config.yaml -o 'finetuned-amazon-product2query' --log_level INFO
    Example configuration file (config.yaml):
    --------------------------------------------------
    data:
        dataset_path: namespace/dataset
        dataset_subset: subset_name
        input_text_column: [column_1, column_2,..]
        label_text_column: target_column
        max_input_length: 512
        max_target_length: 30
        cache_dir: ./cache
        dev: false
    train:
        model_checkpoint: example-t5-base
        output_dir_name: finetuned-model-output
        evaluation_strategy: epoch
        learning_rate: 5e-5
        batch_size: 16
        num_train_epochs: 8
        save_total_limit: 3
        predict_with_generate: true
        logging_strategy: steps
        save_strategy: epoch
        push_to_hub: false
        load_best_model_at_end: true
        metric_for_best_model: eval_loss
        greater_is_better: false
        report_to: none
    --------------------------------------------------
    Outputs:
    - Fine-tuned model saved in the specified directory.
    - Metrics and evaluation results saved in:
        - `evaluation_metrics.csv`: Training and evaluation metrics per step.
        - `final_metrics.csv`: Final best metrics after training completion.
    - Logs stored in `finetuning.log`.
    Logs:
    - Training progress and configuration details are logged for traceability.

"""


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for fine-tuning the model.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser("Finetune model with arguments")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="finetuned-amazon-product2query", help="Model output path"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument("--save_config", action="store_true", help="Save the configuration as YAML for reference")

    return parser.parse_args()



def run_training(args: argparse.Namespace) -> None:
    """Main function to run training pipeline."""
    start_time = datetime.now().astimezone()
    config = Configuration.from_yaml(args.config)

    config.to_yaml(f"{save_path}/config.yaml")  # Save the configuration

    logger.add(f"{save_path}/finetuning.log", level=args.log_level)
    logger.info("=======Starting=======")
    logger.info(f"Configuration saved to '{save_path}/config.yaml'")

    data = config.data
    set_seed(seed=data.seed)
    train = config.train

    model = AutoModelForSeq2SeqLM.from_pretrained(train.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(train.model_checkpoint)

    device = get_device()
    model.to(device=device)

    logger.info(f"Model and Tokenizer: {train.model_checkpoint} is ready.")

    tokenized_datasets = process_data(data, tokenizer)

    # Show the training loss with every epoch
    logging_steps = max(1, len(tokenized_datasets["train"]) // config.train.batch_size)
    logger.info(f"Logging every {logging_steps} steps")

    args = Seq2SeqTrainingArguments(
        output_dir=f"{save_path}/{train.output_dir_name}",
        eval_strategy=train.evaluation_strategy,
        learning_rate=train.learning_rate,
        per_device_train_batch_size=train.batch_size,
        per_device_eval_batch_size=train.batch_size,
        weight_decay=train.weight_decay,
        save_total_limit=train.save_total_limit,
        num_train_epochs=train.num_train_epochs,
        predict_with_generate=train.predict_with_generate,
        logging_steps=logging_steps,
        logging_strategy=train.logging_strategy,
        save_strategy=train.save_strategy,
        push_to_hub=train.push_to_hub,
        load_best_model_at_end=train.load_best_model_at_end,
        metric_for_best_model=train.metric_for_best_model,
        greater_is_better=train.greater_is_better,
        report_to=train.report_to,
    )
    logger.info("Training arguments prepared.")

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        callbacks=[PrinterCallback(save_dir=save_path)],
    )

    logger.info("Trainer initialized. 🚀 Starting training now!(^o^)")

    # Start training
    trainer.train()

    final_output_dir = f"{save_path}/{train.output_dir_name}/final"
    model.save_pretrained(final_output_dir)

    elapsed_time = datetime.now().astimezone() - start_time
    logger.success(f"Training completed in {str(elapsed_time).split('.')[0]} seconds ✅ ^ω^")

if __name__ == "__main__":
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path(f"modules/train/runs/{dt}")

    save_path.mkdir(parents = True, exist_ok = True)

    args = parse_args()
    run_training(args)
