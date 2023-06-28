from pathlib import Path
from typing import List
import typer
from transformers import AutoTokenizer
from data.ordinances import (
    BatchDataModule,
    RandomDataModule,
    WindowDataModule,
)

from utils.utils import tune_batch, tune_incremental


app = typer.Typer()


@app.command()
def batch(
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
    datamodule = BatchDataModule(
        directory=data_directory,
        binarize=binarize,
        tokenizer=tokenizer,
        heuristic_dates=heuristic_dates,
        batch_size=batch_size,
        discard_labels=discard_labels,
        dates_window=dates_window,
    )
    best_value, best_params = tune_batch(datamodule, encoder_model, epochs)
    typer.echo(f"Best metric {best_value} with params {best_params}")


@app.command()
def window(
    data_directory: Path,
    encoder_model: str,
    binarize: bool = False,
    heuristic_dates: bool = False,
    discard_labels: List[str] = None,
    batch_size: int = 16,
    dates_window: int = None,
) -> None:
    discard_labels = set(discard_labels)
    # Loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    # load the dataset first
    datamodule = WindowDataModule(
        directory=data_directory,
        binarize=binarize,
        tokenizer=tokenizer,
        heuristic_dates=heuristic_dates,
        batch_size=batch_size,
        discard_labels=discard_labels,
        dates_window=dates_window,
    )
    best_value, best_params = tune_incremental(datamodule, encoder_model)
    typer.echo(f"Best metric {best_value} with params {best_params}")


@app.command()
def random(
    data_directory: Path,
    encoder_model: str,
    binarize: bool = False,
    heuristic_dates: bool = False,
    discard_labels: List[str] = None,
    batch_size: int = 16,
    dates_window: int = None,
) -> None:
    discard_labels = set(discard_labels)
    # Loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    # load the dataset first
    datamodule = RandomDataModule(
        directory=data_directory,
        binarize=binarize,
        tokenizer=tokenizer,
        heuristic_dates=heuristic_dates,
        batch_size=batch_size,
        discard_labels=discard_labels,
        dates_window=dates_window,
    )
    best_value, best_params = tune_incremental(datamodule, encoder_model)
    typer.echo(f"Best metric {best_value} with params {best_params}")


if __name__ == "__main__":
    app()
