from pathlib import Path


HERE = Path(__file__).resolve().parent

DATA_DIR = HERE / "data"
GZIP_DATA_PATH = DATA_DIR / "rn_data.csv.gz"
DATA_PATH = DATA_DIR / "rn_data.csv"
PREPROCESSED_DATA_PATH = DATA_DIR / "preprocessed_data.txt"
CORRUPT_DATA_PATH = DATA_DIR / "corrupt_data.txt"
TRAIN_DATA_PATH = DATA_DIR / "train_data.txt"
VAL_DATA_PATH = DATA_DIR / "val_data.txt"
TEST_DATA_PATH = DATA_DIR / "test_data.txt"

# Regex to extract the 'target' and 'content' fields
TARGET_AND_CONTENT_PATTERN = r"\[target:(\d+)\].*\[content:'(.*?)'\]"

ENCODING = "utf-8"
