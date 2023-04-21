from pathlib import Path
import time
from typing import List
import typer
from transformers import AutoTokenizer
from data.ordinances import OrdinancesDataModule
from model.ner_model import NERBaseAnnotator
from log import logger

from utils.utils import (
    train_model,
    save_model,
    write_eval_performance,
)


def main(
    data_directory: Path,
    out_dir: Path,
    model_name: str,
    encoder_model: str,
    heuristic_dates: bool = False,
    binarize: bool = False,
    ignore_tags: List[str] = None,
    epochs: int = 16,
    lr: float = 1e-5,
    dropout: float = 0.1,
    weight_decay: float = 0.01,
    grad_norm: float = 1.0,
    batch_size: int = 16,
) -> None:
    ignore_tags = set(ignore_tags)
    timestamp = time.time()
    out_dir_path = out_dir / model_name
    # Loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    # load the dataset first
    datamodule = OrdinancesDataModule(
        data_directory,
        binarize=binarize,
        tokenizer=tokenizer,
        heuristic_dates=heuristic_dates,
        ignore_tags=ignore_tags,
        batch_size=batch_size,
    )
    # Loads the model
    num_training_steps, num_warmup_steps = datamodule.num_training_steps(epochs)
    model = NERBaseAnnotator(
        encoder_model=encoder_model,
        label2id_full=datamodule.label2id_full,
        label2id_filtered=datamodule.label2id_filtered,
        lr=lr,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        dropout_rate=dropout,
        weight_decay=weight_decay,
        stage="training",
    )

    trainer = train_model(
        model=model,
        datamodule=datamodule,
        out_dir=out_dir_path,
        epochs=epochs,
        grad_norm=grad_norm,
    )

    # use pytorch lightnings saver here.
    out_model_path = save_model(
        trainer=trainer,
        out_dir=out_dir_path,
        model_name=model_name,
        timestamp=timestamp,
    )
    logger.info(f"Model saved in {out_model_path}")
    # Evaluates the model on validation and evaluation set
    path = out_dir_path / "val_metrics.tsv"
    out = trainer.test(model=model, dataloaders=datamodule.val_dataloader())
    write_eval_performance(out, path)
    logger.info(f"Validation metrics saved on {path}")
    path = out_dir_path / "eval_metrics.tsv"
    out = trainer.test(model=model, dataloaders=datamodule.eval_dataloader())
    write_eval_performance(out, path)
    logger.info(f"Evaluation metrics saved on {path}")


if __name__ == "__main__":
    typer.run(main)
