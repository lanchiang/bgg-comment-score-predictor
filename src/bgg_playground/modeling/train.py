import argparse
import os

import mlflow
import numpy as np
import torch

from datasets import load_from_disk
from mlflow.models.signature import infer_signature

from bgg_playground.configs.model_config import ModelConfig

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup, AutoTokenizer,
)

from bgg_playground.utils import logs

log = logs.get_logger()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to config file")
    parser.add_argument("--experiment_name", type=str, default="bert-text-classification")
    return parser.parse_args()

def compute_metrics(preds, labels):
    """Calculate accuracy and F1 score."""
    preds = np.argmax(preds, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": accuracy, "f1": f1}

def train(config: ModelConfig):
    # Initialize MLflow
    mlflow.set_experiment(config.experiment_name)
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', default='file:///mlruns'))
    log.debug(f"mlflow tracking server: {os.getenv('MLFLOW_TRACKING_URI')}")

    mlflow.start_run()
    mlflow.log_params(config.__dict__)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    train_dataset = load_from_disk(config.train_data_path)
    val_dataset = load_from_disk(config.val_data_path)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Model
    model = BertForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
    ).to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=len(train_loader) * config.epochs,
    )

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        for batch in val_loader:
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)
                outputs = model(**inputs)

            logits = outputs.logits.detach().cpu().numpy()
            val_preds.extend(logits)
            val_labels.extend(labels.cpu().numpy())

        metrics = compute_metrics(np.array(val_preds), np.array(val_labels))
        mlflow.log_metrics(metrics, step=epoch)

    # Save model and tokenizer
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Log model to MLflow
    signature = infer_signature(
        example_input="Sample input text",
        example_output={"label": 0, "score": 0.99},
    )
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="bert-text-classifier",
        signature=signature,
        input_example="Example input text",
    )

    mlflow.end_run()

if __name__ == "__main__":
    args = parse_args()
    config = ModelConfig.from_yaml(args.config)  # Or load YAML directly
    train(config)