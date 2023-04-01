from pathlib import Path
import time
import typer

from utils.utils import (
    get_reader,
    train_model,
    create_model,
    save_model,
    get_tagset,
)


def main(
    training: Path,
    validation: Path,
    out_dir: Path,
    model_name: str,
    encoder_model: str,
    lr: float,
    epochs: int = 5,
    iob_tagging: str = "kind",
    max_instances: int = -1,
    max_length: int = 512,
    dropout: float = 0.1,
    batch_size: int = 128,
    gpus: int = 1,
    stage: str = "training",
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
    dev_data = get_reader(
        file_path=validation,
        target_vocab=get_tagset(iob_tagging),
        encoder_model=encoder_model,
        max_instances=max_instances,
        max_length=max_length,
    )

    model = create_model(
        train_data=train_data,
        dev_data=dev_data,
        tag_to_id=train_data.get_target_vocab(),
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
