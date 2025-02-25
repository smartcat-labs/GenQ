import os
import csv
import argparse
import yaml
import torch
import nltk
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from evaluate import load
from loguru import logger
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation with a config file.")
    parser.add_argument("config_file", type=str, help="Path to YAML config file")
    return parser.parse_args()

def load_config(config_file: str) -> dict:
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    defaults = {
        "input_text_column": ["title", "description"],
        "label_text_column": "short_query",
        "dataset": "smartcat/Amazon-2023-GenQ",
        "name": None,
        "split": "test",
        "batch_size": 16,
        "model_paths": ["smartcat/T5-product2query-finetune-v1", "BeIR/query-gen-msmarco-t5-base-v1"],
        "sample": 500,
        "seed": 42,
        "log_level": "INFO"
    }

    for key, value in defaults.items():
        if key not in config:
            config[key] = value
            logger.warning(f"Key '{key}' not found in config. Using default: {value}")
    
    return config

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

class RougeScorer:
    def __init__(self):
        self.scorer = load("rouge")
        
    def compute_batch(self, preds, refs):
        try:
            scores = self.scorer.compute(predictions=preds, references=refs, use_aggregator=False)
            return [{
                'rouge1': round(scores['rouge1'][i] * 100, 2),
                'rouge2': round(scores['rouge2'][i] * 100, 2),
                'rougeL': round(scores['rougeL'][i] * 100, 2),
                'rougeLsum': round(scores['rougeLsum'][i] * 100, 2)
            } for i in range(len(preds))]
        except:
            return [{'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0}] * len(preds)

class ModelWrapper:
    def __init__(self, model_path, device):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16 if device.type == 'cuda' else None).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.model.eval()
        
    def generate(self, input_texts):
        with torch.no_grad():
            inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            generated_ids = self.model.generate(
                inputs.input_ids,
                max_length=30,
                num_beams=2,
                early_stopping=True
            )
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

def run_evaluation(config):
    device = get_device()
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(f"eval/runs/{dt}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    rouge_scorer = RougeScorer()
    models = [ModelWrapper(path, device) for path in config['model_paths']]
    
    # Load and prepare dataset
    dataset = load_dataset(config['dataset'], name=config['name'], split=config['split'])
    if config.get('sample'):
        dataset = dataset.shuffle(config['seed']).select(range(config['sample']))
        
    # Preprocess targets once
    dataset = dataset.map(lambda x: {'preprocessed_target': '\n'.join(sent_tokenize(str(x[config['label_text_column']])))},
                          load_from_cache_file=False)

    # Prepare dataloader
    def collate_fn(batch):
        return {
            'input_texts': ['\n\n'.join([str(x[col]) for col in config['input_text_column']]) for x in batch],
            'targets': [x['preprocessed_target'] for x in batch],
            'metadata': [(
                str(x.get('title', '')),
                str(x.get('description', ''))
            ) for x in batch]
        }

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_fn)

    # Prepare CSV
    csv_path = output_path / "results.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'description', 'input_text', 'target', 'model', 
                        'generated', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'])

    # Evaluation loop
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_texts = batch['input_texts']
        targets = batch['targets']
        metadata = batch['metadata']

        rows = []
        for model_wrapper in models:
            generated = model_wrapper.generate(input_texts)
            processed_gen = ['\n'.join(sent_tokenize(g.strip())) for g in generated]
            scores = rouge_scorer.compute_batch(processed_gen, targets)
            
            for (title, desc), inp, tgt, gen, score in zip(metadata, input_texts, targets, generated, scores):
                rows.append([
                    title, desc, inp, tgt,
                    os.path.basename(model_wrapper.tokenizer.name_or_path),
                    gen,
                    score['rouge1'], score['rouge2'], score['rougeL'], score['rougeLsum']
                ])

        # Batch write to CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)

    logger.info(f"Evaluation complete. Results saved to {csv_path}")

def main():
    args = parse_args()
    config = load_config(args.config_file)
    run_evaluation(config)

if __name__ == "__main__":
    main()