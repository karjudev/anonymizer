from pathlib import Path
from typing import List
import typer
from transformers import AutoTokenizer
from data.ordinances import OrdinancesNERDataModule

from utils.utils import tune_model


def main(
    data_directory: Path,
    encoder_model: str,
    binarize: bool = False,
    heuristic_dates: bool = False,
    discard_labels: List[str] = None,
    epochs: int = 3,
    batch_size: int = 16,
    dates_window: int = None,
) -> None:
    discard_labels = set(discard_labels)
    # Loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    # load the dataset first
    datamodule = OrdinancesNERDataModule(
        directory=data_directory,
        binarize=binarize,
        tokenizer=tokenizer,
        heuristic_dates=heuristic_dates,
        batch_size=batch_size,
        discard_labels=discard_labels,
        dates_window=dates_window,
    )
    best_value, best_params = tune_model(datamodule, encoder_model, epochs)
    typer.echo(f"Best metric {best_value} with params {best_params}")


if __name__ == "__main__":
    typer.run(main)
