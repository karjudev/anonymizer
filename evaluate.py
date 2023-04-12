from pathlib import Path
from typing import Iterable, List, Mapping

import typer
import srsly
from transformers import AutoTokenizer, PreTrainedTokenizer
from lightning.pytorch import Trainer

from data.ordinances import OrdinancesDataModule
from utils.utils import load_model, get_out_filename, write_eval_performance
from model.ner_model import NERBaseAnnotator
from model.prediction import predict


def predict_records(
    records: Iterable[Mapping[str, int | str]],
    model: NERBaseAnnotator,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Iterable[Mapping[str, int | str]]:
    for record in records:
        text = record["text"]
        ground_truth = record["entities"]
        predicted = predict(model, tokenizer, text, max_length)
        yield {"text": text, "ground_truth": ground_truth, "predicted": predicted}


def main(
    data_dir: Path,
    model_dir: Path,
    out_dir: Path,
    prefix: str,
    encoder_model: str,
    binarize: bool = False,
    ignore_tags: List[str] = None,
    max_length: int = 512,
    predictions_filename: str = "predictions.jsonl",
) -> None:
    ignore_tags = set(ignore_tags) if len(ignore_tags) > 0 else None
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    datamodule = OrdinancesDataModule(data_dir, binarize, tokenizer, stage="evaluation")
    eval_data = datamodule.eval_dataloader()

    model, model_file = load_model(str(model_dir), tag_to_id=datamodule.tag_to_id)
    trainer = Trainer(accelerator="cpu", enable_checkpointing=False)
    out = trainer.test(model, dataloaders=eval_data)

    eval_file = get_out_filename(out_dir, model_file, prefix=prefix)
    write_eval_performance(out, eval_file)

    pred_output = predict_records(
        srsly.read_jsonl(data_dir / "evaluation.jsonl"), model, tokenizer, max_length
    )
    srsly.write_jsonl(out_dir / predictions_filename, pred_output)


if __name__ == "__main__":
    typer.run(main)
