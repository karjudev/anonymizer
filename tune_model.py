from pathlib import Path
from typing import List
import typer
from transformers import AutoTokenizer
from data.ordinances import OrdinancesDataset

from utils.utils import tune_model


def main(
    training: Path,
    tuning: Path,
    encoder_model: str,
    binarize: bool = False,
    ignore_tags: List[str] = None,
    epochs: int = 3,
    max_length: int = 512,
    dropout: float = 0.1,
    batch_size: int = 16,
) -> None:
    ignore_tags = set(ignore_tags) if len(ignore_tags) > 0 else None
    # Loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    # load the dataset first
    train_data = OrdinancesDataset.from_file(
        filepath=training,
        binarize=binarize,
        tokenizer=tokenizer,
        ignore_tags=ignore_tags,
        max_length=max_length,
    )
    # Gets the start mapping
    label2id = train_data.get_target_vocab()
    # Loads the validation data
    tuning_data = OrdinancesDataset.from_file(
        filepath=tuning,
        binarize=binarize,
        tokenizer=tokenizer,
        ignore_tags=ignore_tags,
        label2id=label2id,
        max_length=max_length,
    )
    best_value, best_params = tune_model(
        train_data, tuning_data, encoder_model, label2id, batch_size, dropout, epochs
    )
    typer.echo(f"Best metric {best_value} with params {best_params}")


if __name__ == "__main__":
    typer.run(main)
