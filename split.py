import os
from pathlib import Path
from typing import List
import typer
import srsly
from sklearn.model_selection import train_test_split


def main(
    jsonl_filepath: Path,
    directory: Path,
    validation_proportion: float = 0.2,
    evaluation_proportion: float = 0.1,
    training_filename: str = "training.jsonl",
    validation_filename: str = "validation.jsonl",
    evaluation_filename: str = "evaluation.jsonl",
    random_state: int = 0,
) -> None:
    records: List = list(srsly.read_jsonl(jsonl_filepath))
    # Holds out evaluation set
    dev, eval = train_test_split(records, test_size=evaluation_proportion, random_state=random_state)
    # Splits training and validation set
    training, validation = train_test_split(dev, test_size=validation_proportion, random_state=random_state)
    # Writes data on disk
    os.makedirs(directory, exist_ok=True)
    srsly.write_jsonl(directory / training_filename, training)
    srsly.write_jsonl(directory / validation_filename, validation)
    srsly.write_jsonl(directory / evaluation_filename, eval)


if __name__ == "__main__":
    typer.run(main)
