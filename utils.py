import datetime
from pathlib import Path

from sklearn.metrics import accuracy_score
import torch


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
GZIP_DATA_PATH = DATA_DIR / "rn_data.csv.gz"
DATA_PATH = DATA_DIR / "rn_data.csv"
PREPROCESSED_DATA_PATH = DATA_DIR / "preprocessed_data.txt"
CORRUPT_DATA_PATH = DATA_DIR / "corrupt_data.txt"
TRAIN_DATA_PATH = DATA_DIR / "train_data.txt"
VAL_DATA_PATH = DATA_DIR / "val_data.txt"
TEST_DATA_PATH = DATA_DIR / "test_data.txt"
CHECKPOINTS_DIR = HERE / "checkpoints"

# Regex to extract the 'target' and 'content' fields
TARGET_AND_CONTENT_PATTERN = r"\[target:(\d+)\].*\[content:'(.*?)'\]"

ENCODING = "utf-8"


def get_datetime():
    """
    Get the current date and time as a formatted string.

    Returns:
        str: The current date and time in the format "YYYY-MM-DD_HH-MM-SS".
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def calculate_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate the accuracy of predictions against the true labels.

    Args:
        preds (torch.Tensor): The predicted labels as a tensor. The
            predictions are expected to be in the form of logits or
            probabilities.
        labels (torch.Tensor): The true labels as a tensor.

    Returns:
        float: The accuracy of the predictions as a float value.
    """
    preds = torch.argmax(preds, dim=1)
    accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    return accuracy


def make_dir(path: Path):
    """
    Creates a directory at the specified path if it does not already exist.

    Parameters:
    path (Path): The path where the directory should be created.

    The function will create any necessary parent directories as well.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)