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
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Evaluation..."):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            total_acc += calculate_accuracy(logits, labels)

    return total_loss / len(data_loader), total_acc / len(data_loader)


def train_step(
    model: BertForSequenceClassification, data_loader: DataLoader, optimizer: AdamW
) -> tuple[float, float]:
    model.train()
    total_loss = 0
    total_acc = 0

    for batch in tqdm(data_loader, total=len(data_loader), desc="Training..."):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

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
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def train_model(
    model: BertForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: AdamW,
    epochs: int = 3,
    run_name: str = "run",
) -> None:
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
    
    
def evaluate_best_model(run_name: str, batch_size: int, device: torch.device) -> None:
    print("Evaluating best model...")

    model = BertForSequenceClassification.from_pretrained(
        CHECKPOINTS_DIR / run_name / "best"
    )
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(CHECKPOINTS_DIR / run_name / "best")
    
    test_samples = 5000
    test_dataset = SentimentDataset(TEST_DATA_PATH, tokenizer, test_samples)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    test_loss, test_acc = eval_step(model, test_dataloader)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}") 
    


if __name__ == "__main__":
    # Used for checkpointing and Tensorboard
    run_name = get_datetime()

    # Ensure the checkpoint directories exist
    make_dir(CHECKPOINTS_DIR / run_name / "best")
    make_dir(CHECKPOINTS_DIR / run_name / "last")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_samples = 10000
    val_samples = 5000
    
    train_dataset = SentimentDataset(TRAIN_DATA_PATH, tokenizer, train_samples)
    val_dataset = SentimentDataset(VAL_DATA_PATH, tokenizer, val_samples)
    

    batch_size = 128

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    train_model(model, train_loader, val_loader, optimizer, 3, run_name)
    
    del train_loader, val_loader, model, tokenizer
    
    evaluate_best_model(run_name, batch_size, device)
