import re
from typing import Optional

from tqdm import tqdm


# Regex to extract the 'target' and 'content' fields
pattern = r"\[target:(\d+)\].*\[content:'(.*?)'\]"


def extract_target_and_content(row: str) -> Optional[tuple[str, str]]:
    """
    Extracts the target and content from a given row string using a regex pattern.

    Args:
        row (str): The input string from which to extract the target and content.

    Returns:
        Optional[tuple[str, str]]: A tuple containing the target and content if the pattern matches,
                                   otherwise None.
    """
    match = re.search(pattern, row)
    if match:
        target = match.group(1)
        content = match.group(2)
        return target, content
    return None


def extact_data_from_rows(data: list[str]) -> list[tuple[str, str]]:
    """
    Extracts the target and content from a list of rows.

    Args:
        data (list[str]): A list of strings from which to extract the target and content.

    Returns:
        list[tuple[str, str]]: A list of tuples containing the target and content.
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


if __name__ == "__main__":
    dataset_path = "data/rn_data.csv"
    # df = pd.read_csv(dataset_path)
    # print(df.head())  # Display the first few rows of the dataset
    
    with open(dataset_path, "r") as f:
        data = f.readlines()
        
    data = extact_data_from_rows(data)
    
    print(data[:5])  # Display the first 5 rows of the extracted data
    