import argparse
import os

import mlflow
import numpy as np
import torch
from datasets import load_from_disk
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,
    DataCollatorWithPadding, pipeline,
)

from bgg_playground.configs.model_config import ModelConfig
from bgg_playground.utils import logs

log = logs.get_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to config file")
    parser.add_argument("--debug_train", type=bool, default=False, help='Whether the workflow runs a debug training job with a subset of dataset')
    parser.add_argument("--experiment_name", type=str, default="bert-text-classification")
    return parser.parse_args()


def compute_metrics(preds, labels):
    """Calculate accuracy and F1 score."""
    preds = np.argmax(preds, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": accuracy, "f1": f1}


def train(config: ModelConfig, **kwargs):
    # Initialize MLflow
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', default='file:///mlruns'))
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(config.__dict__)

        # Device setup
        if torch.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        log.debug(f"Using device: {device}")

        # Load tokenizer and dataset
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

        def tokenize(examples):
            return tokenizer(examples['comment'], padding=True, truncation=True, max_length=512)

        train_dataset = load_from_disk(config.train_data_path).map(tokenize, batched=True).rename_column('rating', 'labels')
        eval_dataset = load_from_disk(config.val_data_path).map(tokenize, batched=True).rename_column('rating', 'labels')

        # Model
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
        ).to(device)

        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            num_train_epochs=config.epochs,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            remove_unused_columns=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
        )

        trainer.train()

        # Save model and tokenizer
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)

        # Create a pipeline for generating signature output
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
        test_input = "This is the best game I have ever played!"
        prediction = classifier(test_input)
        # Log model to MLflow
        signature = infer_signature(
            model_input=test_input,
            model_output=prediction
        )
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="bert-text-classifier",
            signature=signature,
            input_example=test_input,
        )


if __name__ == "__main__":
    args = parse_args()
    config = ModelConfig.from_yaml(args.config)  # Or load YAML directly
    train(config)