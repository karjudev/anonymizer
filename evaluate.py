from pathlib import Path
from typing import List

import typer
from transformers import AutoTokenizer
from lightning.pytorch import Trainer

from data.ordinances import BatchDataModule
from utils.utils import load_model
from log import logger


def main(
    data_dir: Path,
    model_dir: Path,
    encoder_model: str,
    heuristic_dates: bool = False,
    binarize: bool = False,
    discard_labels: List[str] = None,
) -> None:
    discard_labels = set(discard_labels)
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    datamodule = BatchDataModule(
        data_dir,
        binarize,
        tokenizer,
        heuristic_dates,
        discard_labels,
        load_training=False,
    )

    model, _ = load_model(
        str(model_dir),
        eval_label2id=datamodule.eval_label2id,
        training_label2id=datamodule.training_label2id,
    )
    trainer = Trainer(enable_checkpointing=False)
    logger.info("Testing validation set")
    trainer.test(model, dataloaders=datamodule.val_dataloader())
    logger.info("Testing evaluation set")
    trainer.test(model, dataloaders=datamodule.eval_dataloader())


if __name__ == "__main__":
    typer.run(main)
