from dataset import SentimentDataset
from utils import TEST_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def calculate_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1)
    accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    return accuracy


def eval_step(model, data_loader):
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


def train_step(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    total_acc = 0

    for batch in tqdm(data_loader, total=len(data_loader), desc="Training..."):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += calculate_accuracy(logits, labels)

    return total_loss / len(data_loader), total_acc / len(data_loader)
    

def train_model(model, train_loader, val_loader, optimizer, epochs=3):
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_loader, optimizer)
        val_loss, val_acc = eval_step(model, val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    train_dataset = SentimentDataset(TRAIN_DATA_PATH, tokenizer)
    val_dataset = SentimentDataset(VAL_DATA_PATH, tokenizer)
    
    batch_size = 128
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    train_model(model, train_loader, val_loader, optimizer, epochs=3)