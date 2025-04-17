from dataclasses import dataclass

import yaml


@dataclass
class ModelConfig:
    """Configuration for BERT fine-tuning."""
    experiment_name: str
    model_name: str
    tokenizer_path: str
    train_data_path: str
    val_data_path: str
    output_dir: str
    num_labels: int
    learning_rate: float
    epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    weight_decay: float

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """Load config from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)