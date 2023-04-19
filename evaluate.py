from pathlib import Path
from typing import List

import typer
from transformers import AutoTokenizer
from lightning.pytorch import Trainer

from data.ordinances import OrdinancesDataModule
from utils.utils import load_model
from log import logger


def main(
    data_dir: Path,
    model_dir: Path,
    encoder_model: str,
    binarize: bool = False,
    ignore_tags: List[str] = None,
) -> None:
    ignore_tags = set(ignore_tags) if len(ignore_tags) > 0 else None
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    datamodule = OrdinancesDataModule(data_dir, binarize, tokenizer, ignore_tags)

    model, _ = load_model(str(model_dir), label2id=datamodule.label2id)
    trainer = Trainer(enable_checkpointing=False)
    logger.info("Testing validation set")
    trainer.test(model, dataloaders=datamodule.val_dataloader())
    logger.info("Testing evaluation set")
    trainer.test(model, dataloaders=datamodule.eval_dataloader())


if __name__ == "__main__":
    typer.run(main)
