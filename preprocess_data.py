import gzip
from pathlib import Path
import random
import re
import shutil
from typing import Optional

from tqdm import tqdm

from utils import (
    CORRUPT_DATA_PATH,
    ENCODING,
    GZIP_DATA_PATH,
    DATA_PATH,
    PREPROCESSED_DATA_PATH,
    TARGET_AND_CONTENT_PATTERN,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    VAL_DATA_PATH,
)


def extract_target_and_content(row: str) -> Optional[tuple[str, str]]:
    """
    Extracts the target and content from a given row string using a regex
    pattern.

    Args:
        row (str): The input string from which to extract the target and
            content.

    Returns:
        Optional[tuple[str, str]]: A tuple containing the target and content
            if the pattern matches, otherwise None.
    """
    match = re.search(TARGET_AND_CONTENT_PATTERN, row)
    if match:
        target = match.group(1)
        content = match.group(2)
        return target, content
    return None


def extact_data_from_rows(data: list[str]) -> list[tuple[str, str]]:
    """
    Extracts the target and content from a list of rows.

    Args:
        data (list[str]): A list of strings from which to extract the target
            and content.

    Returns:
        list[tuple[str, str]]: A list of tuples containing the target and
            content.
    """
    extracted_data = []
    skipped_rows = 0
    for row in tqdm(data, "Extracting data from rows"):
        target_content = extract_target_and_content(row)
        if target_content:
            extracted_data.append(target_content)
        else:
            print(row)
            skipped_rows += 1
    print(f"Skipped {skipped_rows} rows.")
    return extracted_data


def _strip_special_chars(text: str) -> str:
    """
    Strips special characters from a given text.

    Args:
        text (str): The input text from which to strip special characters.

    Returns:
        str: The text with special characters stripped.
    """
    return text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()


def _preprocess_data(rows: list[str]) -> None:
    """
    Preprocesses a list of data rows by extracting target and content,
    stripping special characters from the content, and writing the
    preprocessed data to a file. Corrupt data rows are written to a
    separate file.

    Args:
        rows (list[str]): A list of data rows to be preprocessed.

    Raises:
        Exception: If an error occurs during preprocessing, it is caught
                   and printed to the console.

    Files:
        Writes preprocessed data to "data/preprocessed_data.txt".
        Writes corrupt data to "data/corrupt_data.txt".
    """

    def _open_file(file_path: str, mode: str = "w"):
        return open(file_path, mode, encoding=ENCODING)

    preprocessed_data = _open_file(PREPROCESSED_DATA_PATH)
    corrupt_data = _open_file(CORRUPT_DATA_PATH)
    try:
        for row in tqdm(rows, "Extracting data from rows"):
            target_content = extract_target_and_content(row)
            if target_content:
                target, content = target_content
                content = _strip_special_chars(content)
                preprocessed_data.write(f"{target} {content}\n")
            else:
                corrupt_data.write(row + "\n")
    except Exception as e:
        print("Error occurred while preprocessing data.")
        print(e)
    finally:
        preprocessed_data.close()
        corrupt_data.close()


def _extract_gzip_file(gzip_file_path: str, output_file_path: str) -> None:
    """
    Extracts a gzip file to a specified output file path.

    Args:
        gzip_file_path (str): The path to the gzip file to be extracted.
        output_file_path (str): The path where the extracted file will be
            saved.
    """
    print(f"Extracting {gzip_file_path} to {output_file_path}")
    with gzip.open(gzip_file_path, "rb") as gz_file:
        with open(output_file_path, "wb") as extracted_file:
            shutil.copyfileobj(gz_file, extracted_file)


def _split_data(data_path: Path, ratios: list[float] = [0.7, 0.2, 0.1]) -> None:
    """
    Splits the data into training, validation, and test sets based on the
        provided ratios.
    Args:
        data_path (Path): The path to the data file.
        ratios (list[float], optional): A list of three floats representing
                                        the ratios for splitting the data
                                        into training, validation, and test
                                        sets. Defaults to [0.7, 0.2, 0.1].
    """
    with open(data_path, "r", encoding=ENCODING) as f:
        data = f.readlines()

    # Split data into positive and negative samples
    positive = []
    negative = []
    for line in data:
        label = line[0]
        if label == "1":
            positive.append(line)
        else:
            negative.append(line)
    num_positive = len(positive)
    num_negative = len(negative)
    print(f"Positive samples total: {num_positive}")
    print(f"Negative samples total: {num_negative}")

    train_ratio, val_ratio, test_ratio = ratios
    num_positive_train = int(num_positive * train_ratio)
    num_negative_train = int(num_negative * train_ratio)
    num_positive_val = int(num_positive * val_ratio)
    num_negative_val = int(num_negative * val_ratio)

    random.shuffle(positive)
    random.shuffle(negative)

    print("Splitting data into train, val and test for ratios: ", ratios)

    train_data = positive[:num_positive_train] + negative[:num_negative_train]
    val_data = (
        positive[num_positive_train : num_positive_train + num_positive_val]
        + negative[num_negative_train : num_negative_train + num_negative_val]
    )
    test_data = (
        positive[num_positive_train + num_positive_val :]
        + negative[num_negative_train + num_negative_val :]
    )

    # Shuffle data so when loading into the model, the data is not ordered in
    # a sens that all positive samples are first and then all negative samples
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    def _save_data(data: list[str], file_path: Path):
        with open(file_path, "w", encoding=ENCODING) as f:
            f.writelines(data)
        print(f"Saved data to {file_path}")

    _save_data(train_data, TRAIN_DATA_PATH)
    _save_data(val_data, VAL_DATA_PATH)
    _save_data(test_data, TEST_DATA_PATH)


def preprocess_data() -> None:
    """
    Preprocesses the data by performing the following steps:
    1. Extracts a gzip file from the specified GZIP_DATA_PATH to DATA_PATH.
    2. Reads the extracted data from DATA_PATH.
    3. Preprocesses the read data.
    4. Prints a message indicating that data preprocessing is complete.
    """
    _extract_gzip_file(GZIP_DATA_PATH, DATA_PATH)
    with open(DATA_PATH, "r") as f:
        data = f.readlines()
    _preprocess_data(data)
    print("Data preprocessing complete.")
    _split_data(PREPROCESSED_DATA_PATH)


if __name__ == "__main__":
    preprocess_data()
