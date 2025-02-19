from dataclasses import dataclass, field
from pathlib import Path
import yaml
from typing import List


@dataclass
class DataConfig:
    """Configuration for dataset and preprocessing"""

    dataset_path: str = "smartcat/Amazon-2023-GenQ"
    dataset_subset: str = ""
    input_text_column: List[str] = field(default_factory=lambda: ["title", "description"]) 
    label_text_column: str = "short_query"
    max_input_length: int = 512
    max_target_length: int = 30
    cache_dir: str = "./cache"
    extend_columns: List[str] = field(default_factory=lambda: ["title"])
    dev: bool = False

def __post_init__(self):
    assert (
        isinstance(self.dataset_path, str) and self.dataset_path
    ), "dataset_path must be a non-empty string"
    assert (
        isinstance(self.dataset_subset, str) and self.dataset_subset
    ), "dataset_subset must be a string"
    assert (
        isinstance(self.input_text_column, List[str]) and self.input_text_column
    ), "input_text_column must be a non-empty list"
    assert (
        isinstance(self.label_text_column, str) and self.label_text_column
    ), "label_text_column must be a non-empty string"
    assert (
        isinstance(self.max_input_length, int) and self.max_input_length > 0
    ), "max_input_length must be a positive integer"
    assert (
        isinstance(self.max_target_length, int) and self.max_target_length > 0
    ), "max_target_length must be a positive integer"
    assert (
        isinstance(self.cache_dir, str) and self.cache_dir
    ), "cache_dir must be a non-empty string"
    assert (
        isinstance(self.extend_columns, bool) and self.extend_columns
    ), "extend_columns must be a non-empty list"
    assert (
        isinstance(self.dev, bool) and self.dev
    ), "dev must be a boolean"



@dataclass
class TrainConfig:
    """Configuration for training arguments"""

    output_dir_name: Path
    model_checkpoint: str = "BeIR/query-gen-msmarco-t5-base-v1"
    metrics: str = "rouge"
    batch_size: int = 8
    num_train_epochs: int = 4
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    learning_rate: float = 5.6e-5
    weight_decay: float = 0.01
    save_total_limit: int = 3
    predict_with_generate: bool = True
    push_to_hub: bool = False
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_rougeL"
    greater_is_better: bool = True
    logging_strategy: str = "epoch"
    report_to: str = "none"

    def __post_init__(self):
        if isinstance(self.output_dir_name, str):
            self.output_dir_name = Path(self.output_dir_name)

        assert (
            self.output_dir_name
        ), "output_dir_name must be a non-empty Path or string"
        assert self.model_checkpoint, "model_checkpoint must be a non-empty string"
        assert self.metrics, "metrics must be a non-empty string"
        assert (
            isinstance(self.batch_size, int) and self.batch_size > 0
        ), "batch_size must be a positive integer"
        assert (
            isinstance(self.num_train_epochs, int) and self.num_train_epochs > 0
        ), "num_train_epochs must be a positive integer"
        assert (
            self.evaluation_strategy
        ), "evaluation_strategy must be a non-empty string"
        assert self.save_strategy, "save_strategy must be a non-empty string"
        assert (
            isinstance(self.learning_rate, (float, int)) and self.learning_rate > 0
        ), "learning_rate must be a positive number"
        assert (
            isinstance(self.weight_decay, (float, int)) and self.weight_decay >= 0
        ), "weight_decay must be a non-negative number"
        assert (
            isinstance(self.save_total_limit, int) and self.save_total_limit > 0
        ), "save_total_limit must be a positive integer"
        assert isinstance(
            self.predict_with_generate, bool
        ), "predict_with_generate must be a boolean"
        assert isinstance(self.push_to_hub, bool), "push_to_hub must be a boolean"
        assert isinstance(
            self.load_best_model_at_end, bool
        ), "load_best_model_at_end must be a boolean"
        assert (
            self.metric_for_best_model
        ), "metric_for_best_model must be a non-empty string"
        assert isinstance(
            self.greater_is_better, bool
        ), "greater_is_better must be a boolean"
        assert self.logging_strategy, "logging_startegy must be a non-empty string"
        assert self.report_to, "report_to must be a non-empty string"


@dataclass
class Configuration:
    """Main configuration class"""

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Configuration":
        """Load configuration from YAML file"""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(
            data=DataConfig(**config_dict["data"]),
            train=TrainConfig(**config_dict["train"]),
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file"""

        def dataclass_to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        config_dict = {
            "data": dataclass_to_dict(self.data),
            "train": dataclass_to_dict(self.train),
        }
        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, sort_keys=False)
