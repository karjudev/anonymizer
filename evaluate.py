from pathlib import Path

import typer
from transformers import AutoTokenizer
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from data.ordinances import BatchDataModule
from utils.utils import load_model
from log import logger


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    data_dir: Path,
    model_dir: Path,
    encoder_model: str,
    binarize: bool = False,
    test: bool = False,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    datamodule = BatchDataModule(data_dir, binarize, tokenizer, load_training=False)

    model, _ = load_model(str(model_dir), label2id=datamodule.label2id)
    trainer = Trainer(enable_checkpointing=False)
    if test:
        logger.info("Testing test set")
        trainer.test(model, dataloaders=datamodule.test_dataloader())
    else:
        logger.info("Testing validation set")
        trainer.test(model, dataloaders=datamodule.val_dataloader())
        logger.info("Testing evaluation set")
        trainer.test(model, dataloaders=datamodule.eval_dataloader())


if __name__ == "__main__":
    app()
