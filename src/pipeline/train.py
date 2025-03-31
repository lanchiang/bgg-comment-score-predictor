import argparse
import os

from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args = parser.parse_args()

    dataset = load_dataset('imdb')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    train_dataset = dataset["train"].map(tokenize, batched=True)
    val_dataset = dataset["test"].map(tokenize, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    # Save model
    trainer.save_model(args.model_dir)


if __name__ == '__main__':
    train()
