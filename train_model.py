from pathlib import Path
import time
from typing import List
import typer
from transformers import AutoTokenizer
from data.ordinances import OrdinancesDataModule
from model.ner_model import NERBaseAnnotator

from utils.utils import (
    train_model,
    save_model,
)


def main(
    data_directory: Path,
    out_dir: Path,
    model_name: str,
    encoder_model: str,
    lr: float,
    binarize: bool = False,
    ignore_tags: List[str] = None,
    epochs: int = 5,
    dropout: float = 0.1,
    batch_size: int = 16,
) -> None:
    ignore_tags = set(ignore_tags) if len(ignore_tags) > 0 else None
    timestamp = time.time()
    out_dir_path = out_dir / model_name
    # Loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    # load the dataset first
    datamodule = OrdinancesDataModule(
        data_directory,
        binarize=binarize,
        tokenizer=tokenizer,
        stage="training",
        ignore_tags=ignore_tags,
        batch_size=batch_size,
    )
    # Loads the model
    num_training_steps, num_warmup_steps = datamodule.num_training_steps(epochs)
    model = NERBaseAnnotator(
        encoder_model=encoder_model,
        tag_to_id=datamodule.tag_to_id,
        lr=lr,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        dropout_rate=dropout,
        stage="training",
    )

    trainer = train_model(
        model=model, datamodule=datamodule, out_dir=out_dir_path, epochs=epochs
    )

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
