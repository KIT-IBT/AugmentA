import os
import csv
from typing import Dict

def load_element_tags(csv_filepath: str = 'Atrial_LDRBM/element_tag.csv') -> Dict[str, str]:
    """
    Loads element tags from a CSV file.
    Returns a dictionary mapping element names to tag values.
    """
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"CSV file {csv_filepath} not found.")

    tag_dict: Dict[str, str] = {}

    try:
        with open(csv_filepath, newline='') as f:
            reader = csv.DictReader(f)

            for row in reader:
                if 'name' not in row or 'tag' not in row:
                    raise ValueError("CSV file missing 'name' or 'tag' columns.")

                tag_dict[row['name']] = row['tag']

    except Exception as e:
        raise RuntimeError(f"Error reading CSV file {csv_filepath}: {e}")

    return tag_dict
