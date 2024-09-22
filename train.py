import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from tqdm import tqdm

from dataset import SentimentDataset
from utils import (
    CHECKPOINTS_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    calculate_accuracy,
    get_datetime,
    make_dir,
)


def eval_step(
    model: BertForSequenceClassification, data_loader: DataLoader
) -> tuple[float, float]:
    """
    Evaluate the performance of a BERT model on a given dataset.
    Args:
        model (BertForSequenceClassification): The BERT model to be evaluated.
        data_loader (DataLoader): DataLoader providing the evaluation dataset.
    Returns:
        tuple[float, float]: A tuple containing the average loss and accuracy
            over the evaluation dataset.
    """
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Evaluation..."):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["label"].to(model.device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            total_acc += calculate_accuracy(logits, labels)

    return total_loss / len(data_loader), total_acc / len(data_loader)


def train_step(
    model: BertForSequenceClassification, data_loader: DataLoader, optimizer: AdamW
) -> tuple[float, float]:
    """
    Perform a single training step for a BERT model.
    Args:
        model (BertForSequenceClassification): The BERT model to be trained.
        data_loader (DataLoader): DataLoader providing the training data.
        optimizer (AdamW): Optimizer for updating the model parameters.
    Returns:
        tuple[float, float]: A tuple containing the average loss and accuracy
            for the training step.
    """
    model.train()
    total_loss = 0
    total_acc = 0

    for batch in tqdm(data_loader, total=len(data_loader), desc="Training..."):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["label"].to(model.device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += calculate_accuracy(logits, labels)

    return total_loss / len(data_loader), total_acc / len(data_loader)


def make_checkpoint(
    model: BertForSequenceClassification, tokenizer: BertTokenizer, checkpoint_dir: Path
) -> None:
    """
    Save the model and tokenizer to the specified directory.

    Args:
        model (BertForSequenceClassification): The BERT model to be saved.
        tokenizer (BertTokenizer): The tokenizer to be saved.
        checkpoint_dir (Path): The directory where the model and tokenizer
            will be saved.
    """
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def train_model(
    model: BertForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: AdamW,
    tokenizer: BertTokenizer,
    epochs: int = 1,
    run_name: str = "run",
) -> None:
    """
    Train a BERT model for sequence classification.
    Args:
        model (BertForSequenceClassification): The BERT model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        optimizer (AdamW): Optimizer for training the model.
        tokenizer (BertTokenizer): Tokenizer used for encoding the text data.
        epochs (int, optional): Number of training epochs. Defaults to 1.
        run_name (str, optional): Name of the training run for logging purposes.
            Defaults to "run".
    """
    tb = SummaryWriter("runs/sentiment_analysis/" + run_name)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_loader, optimizer)
        val_loss, val_acc = eval_step(model, val_loader)

        tb.add_scalar("Loss/train", train_loss, epoch)
        tb.add_scalar("Loss/val", val_loss, epoch)
        tb.add_scalar("Accuracy/train", train_acc, epoch)
        tb.add_scalar("Accuracy/val", val_acc, epoch)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        make_checkpoint(model, tokenizer, CHECKPOINTS_DIR / run_name / "last")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            make_checkpoint(model, tokenizer, CHECKPOINTS_DIR / run_name / "best")

    tb.close()

    print("Training complete!")


def evaluate_best_model(
    run_name: str, batch_size: int, device: torch.device, num_samples: int = -1
) -> None:
    """
    Evaluates the best model for a given run.
    Args:
        run_name (str): The name of the run to evaluate.
        batch_size (int): The batch size to use for evaluation.
        device (torch.device): The device to run the model on (e.g., 'cpu'
            or 'cuda').
        num_samples (int, optional): The number of samples to use for
            evaluation. If -1, use the entire test dataset. Defaults to -1.
    """
    print("Evaluating best model...")

    model = BertForSequenceClassification.from_pretrained(
        CHECKPOINTS_DIR / run_name / "best"
    )
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(CHECKPOINTS_DIR / run_name / "best")

    test_dataset = SentimentDataset(TEST_DATA_PATH, tokenizer, num_samples)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_loss, test_acc = eval_step(model, test_dataloader)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a BERT model for sentiment analysis."
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument(
        "--train_samples", type=int, default=-1, help="Number of training samples."
    )
    parser.add_argument(
        "--val_samples", type=int, default=-1, help="Number of validation samples."
    )
    parser.add_argument(
        "--test_samples", type=int, default=-1, help="Number of test samples."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Used for checkpointing and Tensorboard
    run_name = get_datetime()

    # Ensure the checkpoint directories exist
    make_dir(CHECKPOINTS_DIR / run_name / "best")
    make_dir(CHECKPOINTS_DIR / run_name / "last")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = SentimentDataset(TRAIN_DATA_PATH, tokenizer, args.train_samples)
    val_dataset = SentimentDataset(VAL_DATA_PATH, tokenizer, args.val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    train_model(
        model, train_loader, val_loader, optimizer, tokenizer, args.epochs, run_name
    )

    del train_loader, val_loader, model, tokenizer

    evaluate_best_model(run_name, args.batch_size, device, args.test_samples)


if __name__ == "__main__":
    main()
    # python train.py --epochs 3 --train_samples 100000 --val_samples 10000 --test_samples 10000 --batch_size 128