import os
import csv
from typing import Dict


class TagLoader:
    """
    Loads element tags from a specified CSV file.
    Encapsulates the logic for reading and parsing the tag mapping.
    """

    def __init__(self, csv_filepath: str):
        if not os.path.exists(csv_filepath):
            raise FileNotFoundError(f"Tag CSV file not found: {csv_filepath}")
        self.filepath = csv_filepath

    def load(self) -> Dict[str, str]:
        """
        Reads the CSV file and returns a dictionary mapping element names to tags.
        """
        tag_dict: Dict[str, str] = {}
        try:
            with open(self.filepath, newline='') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    if 'name' not in row or 'tag' not in row:
                        raise ValueError("CSV file missing 'name' or 'tag' columns.")

                    tag_dict[row['name']] = row['tag']
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file {self.filepath}: {e}")

        return tag_dict
