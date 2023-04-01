from pathlib import Path

from torch.utils.data import DataLoader
import typer

from utils.utils import (
    get_reader,
    load_model,
    get_trainer,
    get_out_filename,
    write_eval_performance,
    get_tagset,
)


def main(
    test: Path,
    model_path: Path,
    out_dir: Path,
    prefix: str,
    encoder_model: str,
    iob_tagging: str = "kind",
    max_instances: int = -1,
    max_length: int = 512,
    batch_size: int = 128,
) -> None:
    # load the dataset first
    test_data = get_reader(
        file_path=test,
        target_vocab=get_tagset(iob_tagging),
        max_instances=max_instances,
        max_length=max_length,
        encoder_model=encoder_model,
    )

    model, model_file = load_model(model_path, tag_to_id=get_tagset(iob_tagging))
    trainer = get_trainer(is_test=True)
    out = trainer.test(
        model,
        test_dataloaders=DataLoader(
            test_data, batch_size=batch_size, collate_fn=model.collate_batch
        ),
    )

    # use pytorch lightnings saver here.
    eval_file = get_out_filename(out_dir, model_file, prefix=prefix)
    write_eval_performance(out, eval_file)


if __name__ == "__main__":
    typer.run(main)
