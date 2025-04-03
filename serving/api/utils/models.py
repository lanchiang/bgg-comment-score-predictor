import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from .logs import get_logger

log = get_logger()


class ModelContainer:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None

    def load_model(self, model_name: str = "bert-base-uncased", num_labels: int = 2):
        """Initialize model components"""
        model_name = os.getenv(key='MODEL_NAME', default='bert-base-uncased')
        num_labels = os.getenv(key='NUM_LABELS', default=2)

        if torch.backends.cuda.is_built() and torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_built() and torch.mps.is_available():
            device = torch.device("mps")
        else:
            device = "cpu"
        log.info(f"Using device: {device}")

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        log.info("Model and tokenizer loaded successfully")

        self.model.eval()

    def cleanup(self):
        log.info("Cleaning up resources...")
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()
        elif self.device == torch.device("mps"):
            torch.mps.empty_cache()


# Singleton instance
model_container = ModelContainer()