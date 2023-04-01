import os
from pathlib import Path
from typing import List
import typer
import srsly


def main(
    jsonl_filepath: Path,
    directory: Path,
    validation_proportion: float = 0.2,
    tuning_proportion: float = 0.2,
    training_filename: str = "training.jsonl",
    tuning_filename: str = "tuning.jsonl",
    validation_filename: str = "validation.jsonl",
) -> None:
    records: List = list(srsly.read_jsonl(jsonl_filepath))
    # Number of records in the validation set
    num_validation = int(len(records) * validation_proportion)
    # Extracts the validation set
    dev, validation = records[:-num_validation], records[-num_validation:]
    # Number of records in the tuning set
    num_tuning = int(len(dev) * tuning_proportion)
    # Extracts the tuning set
    training, tuning = dev[:-num_tuning], dev[-num_tuning:]
    # Writes data on disk
    os.makedirs(directory, exist_ok=True)
    srsly.write_jsonl(directory / training_filename, training)
    srsly.write_jsonl(directory / tuning_filename, tuning)
    srsly.write_jsonl(directory / validation_filename, validation)


if __name__ == "__main__":
    typer.run(main)
