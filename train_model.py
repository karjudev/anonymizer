from enum import Enum
from pathlib import Path
import time
import typer

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from active_learning.query import distance_query, entropy_query, random_query
from active_learning.training import active_learning_loop, batch_loop

from log import logger
from active_learning.dataset import ActiveLearningDataset
from data.ordinances import (
    BatchDataModule,
    OrdinancesNERDataset,
    get_collate_function,
    training_warmup_steps,
)
from model.ner_model import NERBaseAnnotator

from utils.utils import (
    train_batch,
    save_model,
    write_eval_performance,
)

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def batch(
    data_directory: Path,
    out_dir: Path,
    model_name: str,
    encoder_model: str,
    binarize: bool = False,
    epochs: int = 16,
    lr: float = 1e-5,
    dropout: float = 0.1,
    weight_decay: float = 0.01,
    grad_norm: float = 1.0,
    batch_size: int = 16,
) -> None:
    timestamp = time.time()
    out_dir_path = out_dir / model_name
    # Loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    # load the dataset first
    datamodule = BatchDataModule(
        data_directory,
        binarize=binarize,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    # Loads the model
    num_training_steps, num_warmup_steps = training_warmup_steps(
        dataset=datamodule.training,
        batch_size=batch_size,
        num_gpus=datamodule.num_gpus,
        epochs=epochs,
    )
    model = NERBaseAnnotator(
        encoder_model=encoder_model,
        label2id=datamodule.label2id,
        lr=lr,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        dropout_rate=dropout,
        weight_decay=weight_decay,
        stage="training",
    )
    # Trains the model in batch mode
    trainer = batch_loop(
        out_dir=out_dir,
        model=model,
        training_dataloader=datamodule.train_dataloader(),
        val_dataloader=datamodule.val_dataloader(),
        grad_norm=grad_norm,
        epochs=epochs,
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


class QueryStrategyType(Enum):
    RANDOM = "random"
    ENTROPY = "entropy"
    DISTANCE = "distance"


@app.command()
def active(
    data_directory: Path,
    out_dir: Path,
    model_name: str,
    encoder_model: str,
    binarize: bool = False,
    strategy: QueryStrategyType = "random",
    num_queries: int = 30,
    query_size: int = 16,
    epochs: int = 4,
    lr: float = 1e-5,
    dropout: float = 0.1,
    weight_decay: float = 0.01,
    grad_norm: float = 1.0,
    batch_size: int = 16,
    training_filename: str = "training.jsonl",
    validation_filename: str = "validation.jsonl",
    evaluation_filename: str = "evaluation.jsonl",
) -> None:
    timestamp = time.time()
    out_dir_path = out_dir / model_name
    # Loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    # Loads the training data in active learning mode
    training = ActiveLearningDataset(
        OrdinancesNERDataset.from_file(
            filepath=data_directory / training_filename,
            binarize=binarize,
            tokenizer=tokenizer,
        )
    )
    # Extracts the collate function
    collate_training = get_collate_function(
        tokenizer, training.label2id, active_learning=True
    )
    collate_validation = get_collate_function(
        tokenizer, training.label2id, active_learning=False
    )
    # Gets the query strategy
    if strategy == QueryStrategyType.RANDOM:
        query_strategy = random_query
    elif strategy == QueryStrategyType.ENTROPY:
        query_strategy = entropy_query
    elif strategy == QueryStrategyType.DISTANCE:
        query_strategy = distance_query
    else:
        raise NotImplementedError
    # Loads the validation dataloader
    validation = DataLoader(
        OrdinancesNERDataset.from_file(
            filepath=data_directory / validation_filename,
            binarize=binarize,
            tokenizer=tokenizer,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_validation,
    )
    # Loads evaluation dataloader
    evaluation = DataLoader(
        OrdinancesNERDataset.from_file(
            filepath=data_directory / evaluation_filename,
            binarize=binarize,
            tokenizer=tokenizer,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_validation,
    )
    # Loads the model
    num_training_steps, num_warmup_steps = training_warmup_steps(
        dataset=training,
        batch_size=batch_size,
        num_gpus=1,
        epochs=epochs,
    )
    model = NERBaseAnnotator(
        encoder_model=encoder_model,
        label2id=training.label2id,
        lr=lr,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        dropout_rate=dropout,
        weight_decay=weight_decay,
        stage="training",
    )
    # Runs the active training loop
    trainer = active_learning_loop(
        out_dir_path,
        model,
        training,
        validation,
        collate_fn=collate_training,
        query_strategy=query_strategy,
        query_size=query_size,
        num_queries=num_queries,
        grad_norm=grad_norm,
        epochs=epochs,
        batch_size=batch_size,
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
    out = trainer.test(model=model, dataloaders=validation)
    write_eval_performance(out, path)
    logger.info(f"Validation metrics saved on {path}")
    path = out_dir_path / "eval_metrics.tsv"
    out = trainer.test(model=model, dataloaders=evaluation)
    write_eval_performance(out, path)
    logger.info(f"Evaluation metrics saved on {path}")


if __name__ == "__main__":
    app()
