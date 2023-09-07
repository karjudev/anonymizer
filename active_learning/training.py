import json
import os
from pathlib import Path
from typing import Callable, List
from lightning import seed_everything
from torch.utils.data import DataLoader
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping

from active_learning.dataset import ActiveLearningDataset
from active_learning.query import QueryStrategy, query_the_oracle


def batch_loop(
    out_dir: Path,
    model: LightningModule,
    training_dataloader: DataLoader,
    val_dataloader: DataLoader,
    grad_norm: float,
    epochs: int,
    metric: str = "val_MD@F1",
    callbacks: List[Callable] = [],
    log_filename: str = "metrics.jsonl",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(42)
    for _ in range(epochs):
        # Trains the model for the given number of epochs
        trainer = Trainer(
            max_epochs=1,
            gradient_clip_algorithm="norm",
            gradient_clip_val=grad_norm,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=out_dir,
            callbacks=callbacks,
        )
        trainer.fit(
            model=model,
            train_dataloaders=training_dataloader,
            val_dataloaders=val_dataloader,
        )
        metrics = trainer.test(model=model, dataloaders=val_dataloader, verbose=False)[
            0
        ]
        with open(out_dir / log_filename, mode="a") as file:
            file.write(json.dumps(metrics) + "\n")
    return trainer


def active_learning_loop(
    out_dir: Path,
    model: LightningModule,
    training_dataset: ActiveLearningDataset,
    val_dataloader: DataLoader,
    collate_fn: Callable,
    query_strategy: QueryStrategy,
    query_size: int,
    num_queries: int,
    grad_norm: float,
    epochs: int,
    batch_size: int,
    callbacks: List[Callable] = [],
    log_filename: str = "metrics.jsonl",
) -> Trainer:
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(42)
    for _ in range(num_queries):
        # Queries the oracle and gets more labels
        sample_idx = query_the_oracle(
            model,
            training_dataset,
            collate_fn=collate_fn,
            query_size=query_size,
            query_strategy=query_strategy,
            batch_size=batch_size,
        )
        training_dataset.label(sample_idx)
        # Extracts all the labelled samples
        labelled_loader = training_dataset.get_labelled(
            batch_size=batch_size, collate_fn=collate_fn
        )
        # Trains the model for a given number of epochs
        trainer = Trainer(
            max_epochs=epochs,
            gradient_clip_algorithm="norm",
            gradient_clip_val=grad_norm,
            check_val_every_n_epoch=epochs,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=out_dir,
            callbacks=callbacks,
        )
        trainer.fit(model=model, train_dataloaders=labelled_loader)
        metrics = trainer.test(model=model, dataloaders=val_dataloader, verbose=False)[
            0
        ]
        with open(out_dir / log_filename, mode="a") as file:
            file.write(json.dumps(metrics) + "\n")
    return trainer
