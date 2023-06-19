from pathlib import Path
import time
from typing import List
import typer
from transformers import AutoTokenizer
from data.ordinances import BatchDataModule, IncrementalDataModule
from model.ner_model import NERBaseAnnotator
from log import logger

from utils.utils import (
    train_batch,
    save_model,
    train_incremental,
    write_eval_performance,
)

app = typer.Typer()


@app.command()
def batch(
    data_directory: Path,
    out_dir: Path,
    model_name: str,
    encoder_model: str,
    heuristic_dates: bool = False,
    binarize: bool = False,
    discard_labels: List[str] = None,
    dates_window: int = None,
    epochs: int = 16,
    lr: float = 1e-5,
    dropout: float = 0.1,
    weight_decay: float = 0.01,
    grad_norm: float = 1.0,
    batch_size: int = 16,
) -> None:
    discard_labels = set(discard_labels)
    timestamp = time.time()
    out_dir_path = out_dir / model_name
    # Loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    # load the dataset first
    datamodule = BatchDataModule(
        data_directory,
        binarize=binarize,
        tokenizer=tokenizer,
        heuristic_dates=heuristic_dates,
        discard_labels=discard_labels,
        dates_window=dates_window,
        batch_size=batch_size,
    )
    # Loads the model
    num_training_steps, num_warmup_steps = datamodule.training_warmup_steps(epochs)
    model = NERBaseAnnotator(
        encoder_model=encoder_model,
        eval_label2id=datamodule.eval_label2id,
        training_label2id=datamodule.training_label2id,
        lr=lr,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        dropout_rate=dropout,
        weight_decay=weight_decay,
        stage="training",
    )

    trainer = train_batch(
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


@app.command()
def incremental(
    data_filepath: Path,
    out_dir: Path,
    model_name: str,
    encoder_model: str,
    heuristic_dates: bool = False,
    binarize: bool = False,
    discard_labels: List[str] = None,
    dates_window: int = None,
    lr: float = 1e-5,
    dropout: float = 0.1,
    weight_decay: float = 0.01,
    grad_norm: float = 1.0,
    batch_size: int = 16,
) -> None:
    discard_labels = set(discard_labels)
    timestamp = time.time()
    out_dir_path = out_dir / model_name
    # Loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    # load the dataset first
    datamodule = IncrementalDataModule(
        data_filepath,
        binarize=binarize,
        tokenizer=tokenizer,
        heuristic_dates=heuristic_dates,
        discard_labels=discard_labels,
        dates_window=dates_window,
        batch_size=batch_size,
    )
    # Automatically computes the epochs
    epochs = len(datamodule.dataset) // datamodule.batch_size
    # Loads the model
    num_training_steps, num_warmup_steps = datamodule.training_warmup_steps(epochs)
    model = NERBaseAnnotator(
        encoder_model=encoder_model,
        eval_label2id=datamodule.eval_label2id,
        training_label2id=datamodule.training_label2id,
        lr=lr,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        dropout_rate=dropout,
        weight_decay=weight_decay,
        stage="training",
    )

    trainer = train_incremental(
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
    app()
