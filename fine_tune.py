from pathlib import Path
import time

import typer

from utils.utils import (
    get_reader,
    train_model,
    save_model,
    get_tagset,
    load_model,
)


def main(
    training: Path,
    model_path: Path,
    model_name: str,
    out_dir: Path,
    encoder_model: str,
    epochs: int = 5,
    iob_tagging: str = "kind",
    max_instances: int = -1,
    max_length: int = 512,
) -> None:
    timestamp = time.time()
    out_dir_path = out_dir / model_name

    # load the dataset first
    train_data = get_reader(
        file_path=training,
        target_vocab=get_tagset(iob_tagging),
        encoder_model=encoder_model,
        max_instances=max_instances,
        max_length=max_length,
    )
    model, model_file = load_model(
        model_path, label2id=get_tagset(iob_tagging), stage="finetune"
    )
    model.train_data = train_data

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
