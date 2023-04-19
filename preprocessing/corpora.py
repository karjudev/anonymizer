import os
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Set
from sklearn.model_selection import train_test_split
import typer
import srsly


ACCEPTED_LABELS = {"MISC", "ORG", "PER", "LOC", "TIME"}


app = typer.Typer()


def _read_wikiner(path: Path) -> Iterator[Mapping[str, List[str]]]:
    with open(path) as file:
        for line in file:
            tokens, labels = [], []
            for token_pos_ner in line.split():
                parts = token_pos_ner.split("|")
                token, label = parts[0], parts[-1]
                tokens.append(token)
                labels.append(label)
            yield {"tokens": tokens, "labels": labels}


def _filter_labels(
    records: Iterable[Mapping[str, List[str]]], labels: Set[str], out_label: str = "O"
) -> Iterator[Mapping[str, List[str]]]:
    for record in records:
        for i in range(len(record["labels"])):
            label = record["labels"][i]
            if label[2:] not in labels:
                record["labels"][i] = out_label
        yield record


@app.command()
def wikiner(
    filepath: Path,
    out_dir: Path,
    accepted_labels: List[str] = None,
    validation_proportion: float = 0.2,
    training_filename: str = "training.jsonl",
    validation_filename: str = "validation.jsonl",
    random_state: int = 0,
) -> None:
    if len(accepted_labels) == 0:
        accepted_labels = ACCEPTED_LABELS
    else:
        accepted_labels = set(accepted_labels)
    # Reads and parses the WikiNER records from file
    records = list(_read_wikiner(filepath))
    # Filters the accepted labels
    records = _filter_labels(records, accepted_labels)
    # Splits them random into training and validation
    training, validation = train_test_split(
        records, test_size=validation_proportion, random_state=random_state
    )
    # Writes data on disk
    os.makedirs(out_dir, exist_ok=True)
    srsly.write_jsonl(out_dir / training_filename, training)
    srsly.write_jsonl(out_dir / validation_filename, validation)


def _read_tsv(
    path: Path, token_idx: int, label_idx: int
) -> Iterator[Mapping[str, List[str]]]:
    with open(path) as file:
        tokens, labels = [], []
        for line in file:
            parts = line.split()
            if len(parts) == 0:
                yield {"tokens": tokens, "labels": labels}
                tokens, labels = [], []
            else:
                token, label = parts[token_idx], parts[label_idx]
                tokens.append(token)
                labels.append(label)


@app.command()
def multinerd(
    filepath: Path,
    out_dir: Path,
    accepted_labels: List[str] = None,
    validation_proportion: float = 0.2,
    training_filename: str = "training.jsonl",
    validation_filename: str = "validation.jsonl",
    random_state: int = 0,
) -> None:
    if len(accepted_labels) == 0:
        accepted_labels = ACCEPTED_LABELS
    else:
        accepted_labels = set(accepted_labels)
    # Reads and parses the WikiNER records from file
    records = _read_tsv(filepath, token_idx=1, label_idx=2)
    # Optionally filters labels
    records = list(_filter_labels(records, accepted_labels))
    # Splits them random into training and validation
    training, validation = train_test_split(
        records, test_size=validation_proportion, random_state=random_state
    )
    # Writes data on disk
    os.makedirs(out_dir, exist_ok=True)
    srsly.write_jsonl(out_dir / training_filename, training)
    srsly.write_jsonl(out_dir / validation_filename, validation)


def _build_kind_out_filepath(filepath: Path, out_dir: Path) -> Path:
    filename = filepath.name
    filename = filename[: filename.rindex(".")]
    out_filepath = out_dir / (filename + ".jsonl")
    return out_filepath


def _add_iob_tagging(
    records: Iterable[Mapping[str, List[str]]], out_label: str = "O"
) -> Iterator[Mapping[str, List[str]]]:
    for record in records:
        in_entity = False
        labels = record["labels"]
        for i in range(len(labels)):
            label = labels[i]
            if label != out_label:
                prefix = "I" if in_entity else "B"
                labels[i] = f"{prefix}-{label}"
                in_entity = True
            else:
                in_entity = False
        record["labels"] = labels
        yield record


def _process_kind_file(
    filepath: Path, out_dir: Path, accepted_labels: Set[str]
) -> None:
    # Creates the output filepath
    out_filepath = _build_kind_out_filepath(filepath, out_dir)
    # Reads the records
    records = _read_tsv(filepath, token_idx=0, label_idx=1)
    # Adds IOB tagging
    records = _add_iob_tagging(records)
    # Filters out the records
    records = _filter_labels(records, accepted_labels)
    # Serializes the records
    srsly.write_jsonl(out_filepath, records)


@app.command()
def kind_file(filepath: Path, out_dir: Path, accepted_labels: List[str] = None) -> None:
    # Sets the set of accepted labels
    if len(accepted_labels) == 0:
        accepted_labels = ACCEPTED_LABELS
    else:
        accepted_labels = set(accepted_labels)
    os.makedirs(out_dir, exist_ok=True)
    _process_kind_file(filepath, out_dir, accepted_labels)


@app.command()
def kind(dir: Path, out_dir: Path, accepted_labels: List[str] = None) -> None:
    # Sets the set of accepted labels
    if len(accepted_labels) == 0:
        accepted_labels = ACCEPTED_LABELS
    else:
        accepted_labels = set(accepted_labels)
    os.makedirs(out_dir, exist_ok=True)
    # Analyzes each file
    for entry in os.scandir(dir):
        if entry.is_file() and entry.name.endswith("tsv"):
            _process_kind_file(Path(entry.path), out_dir, accepted_labels)


if __name__ == "__main__":
    app()
