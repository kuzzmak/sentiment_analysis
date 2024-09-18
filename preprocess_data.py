import gzip
import re
import shutil
from typing import Optional

from tqdm import tqdm

from utils import GZIP_DATA_PATH, DATA_PATH, TARGET_AND_CONTENT_PATTERN


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
    for row in tqdm(data, 'Extracting data from rows'):
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
    return text.replace("\n", " ") \
        .replace("\r", " ") \
        .replace("\t", " ") \
        .strip()
        

def _open_file(file_path: str, mode: str = "w"):
    """
    Opens a file with the specified mode and UTF-8 encoding.

    Args:
        file_path (str): The path to the file to be opened.
        mode (str, optional): The mode in which to open the file. Defaults
            to "w".

    Returns:
        file object: The opened file object.
    """
    return open(file_path, mode, encoding="utf-8")


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
    preprocessed_data = _open_file("data/preprocessed_data.txt")
    corrupt_data = _open_file("data/corrupt_data.txt")
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


if __name__ == "__main__":  
    preprocess_data()
    