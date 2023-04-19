import os
from pathlib import Path
from typing import Iterator, List, Mapping
import typer
import srsly
from sklearn.model_selection import train_test_split


app = typer.Typer()


@app.command()
def split(
    jsonl_filepath: Path,
    directory: Path,
    validation_proportion: float = 0.2,
    evaluation_proportion: float = 0.1,
    training_filename: str = "training.jsonl",
    validation_filename: str = "validation.jsonl",
    evaluation_filename: str = "evaluation.jsonl",
    random_state: int = 0,
) -> None:
    records: List = list(srsly.read_jsonl(jsonl_filepath))
    # Holds out evaluation set
    dev, eval = train_test_split(
        records, test_size=evaluation_proportion, random_state=random_state
    )
    # Splits training and validation set
    training, validation = train_test_split(
        dev, test_size=validation_proportion, random_state=random_state
    )
    # Writes data on disk
    os.makedirs(directory, exist_ok=True)
    srsly.write_jsonl(directory / training_filename, training)
    srsly.write_jsonl(directory / validation_filename, validation)
    srsly.write_jsonl(directory / evaluation_filename, eval)


def _can_merge(
    first: Mapping[str, int | str], second: Mapping[str, int | str], text: str
) -> bool:
    if first["label"] != second["label"]:
        return False
    left, right = first["end"], second["start"]
    return text[left:right].isspace()


def merge_contiguous_spans(record: Mapping[str, int | str]) -> Mapping[str, int | str]:
    text = record["text"]
    spans = sorted(record["entities"], key=lambda span: span["start"])
    merged_spans = []
    for span in spans:
        if len(merged_spans) == 0 or not _can_merge(merged_spans[-1], span, text):
            merged_spans.append(span)
        else:
            merged_spans[-1]["end"] = span["end"]
    typer.echo(f"Old spans:\t{len(spans)}\tMerged spans:\t{len(merged_spans)}")
    record["entities"] = merged_spans
    return record


def strip_alphanumeric_spans(
    record: Mapping[str, int | str]
) -> Mapping[str, int | str]:
    text = record["text"]
    spans = record["entities"]
    filtered_spans = []
    for span in spans:
        start, end = span["start"], span["end"]
        # Advances start pointer
        while start < end and not text[start].isalnum():
            start += 1
        # Pulls back end pointer
        while end > start and not text[end - 1].isalnum():
            end -= 1
        if end - start > 0:
            filtered_spans.append({"start": start, "end": end, "label": span["label"]})
    typer.echo(f"Old spans:\t{len(spans)}\tFiltered spans:\t{len(filtered_spans)}")
    record["entities"] = filtered_spans
    return record


def clean_records(
    records: List[Mapping[str, int | str]], strip_alphanum: bool, merge_contiguous: bool
) -> Iterator[Mapping[str, int | str]]:
    for record in records:
        if strip_alphanum:
            record = strip_alphanumeric_spans(record)
        if merge_contiguous:
            record = merge_contiguous_spans(record)
        yield record


@app.command()
def clean(
    in_filepath: Path,
    out_filepath: Path,
    strip_alphanum: bool = False,
    merge_contiguous: bool = False,
) -> None:
    assert str(in_filepath.absolute()) != str(out_filepath.absolute())
    input_stream = srsly.read_jsonl(in_filepath)
    output_stream = clean_records(input_stream, strip_alphanum, merge_contiguous)
    srsly.write_jsonl(out_filepath, output_stream)


if __name__ == "__main__":
    app()
