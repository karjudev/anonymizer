from pathlib import Path
import time
from typing import List
import typer
from transformers import AutoTokenizer
from data.ordinances import OrdinancesDataset

from utils.utils import (
    train_model,
    create_model,
    save_model,
)


def main(
    training: Path,
    validation: Path,
    out_dir: Path,
    model_name: str,
    encoder_model: str,
    lr: float,
    binarize: bool = False,
    ignore_tags: List[str] = None,
    epochs: int = 5,
    max_length: int = 512,
    dropout: float = 0.1,
    batch_size: int = 16,
    gpus: int = 1,
    stage: str = "training",
) -> None:
    ignore_tags = set(ignore_tags) if len(ignore_tags) > 0 else None
    timestamp = time.time()
    out_dir_path = out_dir / model_name
    # Loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    # load the dataset first
    train_data = OrdinancesDataset.from_file(
        filepath=training,
        binarize=binarize,
        tokenizer=tokenizer,
        ignore_tags=ignore_tags,
        max_length=max_length
    )
    # Gets the start mapping
    label2id = train_data.get_target_vocab()
    # Loads the validation data
    dev_data = OrdinancesDataset.from_file(
        filepath=validation,
        binarize=binarize,
        tokenizer=tokenizer,
        ignore_tags=ignore_tags,
        label2id=label2id,
        max_length=max_length
    )

    model = create_model(
        train_data=train_data,
        dev_data=dev_data,
        tag_to_id=label2id,
        dropout_rate=dropout,
        batch_size=batch_size,
        stage=stage,
        lr=lr,
        encoder_model=encoder_model,
        num_gpus=gpus,
    )

    trainer = train_model(model=model, out_dir=out_dir_path, epochs=epochs)

    # use pytorch lightnings saver here.
    out_model_path = save_model(
        trainer=trainer,
        out_dir=out_dir_path,
        model_name=model_name,
        timestamp=timestamp,
    )
    typer.echo(f"Model saved in {out_model_path}")


if __name__ == "__main__":

    typer.run(main)
