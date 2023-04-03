from pathlib import Path
from typing import List
import typer
from transformers import AutoTokenizer
from data.ordinances import OrdinancesDataset, OrdinancesDataModule

from utils.utils import tune_model


def main(
    data_directory: Path,
    encoder_model: str,
    binarize: bool = False,
    ignore_tags: List[str] = None,
    epochs: int = 3,
    dropout: float = 0.1,
    batch_size: int = 16,
) -> None:
    ignore_tags = set(ignore_tags) if len(ignore_tags) > 0 else None
    # Loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    # load the dataset first
    datamodule = OrdinancesDataModule(
        directory=data_directory,
        binarize=binarize,
        tokenizer=tokenizer,
        stage="tuning",
        batch_size=batch_size,
        ignore_tags=ignore_tags,
    )
    best_value, best_params = tune_model(datamodule, encoder_model, epochs, dropout)
    typer.echo(f"Best metric {best_value} with params {best_params}")


if __name__ == "__main__":
    typer.run(main)
