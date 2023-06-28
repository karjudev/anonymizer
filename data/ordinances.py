from abc import abstractmethod
import os
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Set, Tuple
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS

import torch
from heuristics.date import detect_dates
from log import logger
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from lightning.pytorch import LightningDataModule
import srsly
from transformers import PreTrainedTokenizer
from datasets import DatasetDict, load_dataset

from data.spans import (
    extract_spans,
    get_label2id,
    discard_label2id,
    prodigy_to_labels,
)


# Disable parallel tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Example used for prediction and learning
Example = Tuple[
    torch.Tensor,
    torch.Tensor,
    List[int],
    Mapping[Tuple[int, int], str],
    torch.Tensor,
]


def encode_text(
    text: str, tokenizer: PreTrainedTokenizer, max_length: int
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]]:
    output = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=False,
        return_attention_mask=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    for input_ids, attention_mask, offset_mapping in zip(
        output["input_ids"], output["attention_mask"], output["offset_mapping"]
    ):
        yield input_ids, attention_mask.type(torch.bool), offset_mapping


def _binarize(
    entities: Mapping[str, int | str], label: str = "OMISSIS"
) -> Mapping[str, int | str]:
    for entity in entities:
        entity["label"] = label
    return entities


def _discard(
    entities: Mapping[str, int | str], tags: Set[str]
) -> Mapping[str, int | str]:
    filtered_entities = []
    for entity in entities:
        if entity["label"] not in tags:
            filtered_entities.append(entity)
    return filtered_entities


def _discard_tags(
    records: List[Mapping[str, int | str]], tags: Set[str]
) -> List[Mapping[int, int | str]]:
    for record in records:
        record["entities"] = _discard(record["entities"], tags)
    return records


class OrdinancesNERDataset(Dataset):
    def __init__(
        self,
        binarize: bool,
        tokenizer: PreTrainedTokenizer,
        heuristic_dates: bool,
        discard_labels: Set[str],
        max_length: int = 512,
        dates_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.__binarize = binarize
        self.__heuristic_dates = heuristic_dates
        self.__discard_labels = discard_labels
        # Set of labels handled by the heuristics and non the ML model
        self.__filter_labels = set()
        if heuristic_dates:
            self.__filter_labels.add("TIME")

        self.eval_label2id = discard_label2id(
            get_label2id(binarize), self.__discard_labels
        )
        self.__eval_id2label = {idx: label for label, idx in self.eval_label2id.items()}
        self.training_label2id = discard_label2id(
            self.eval_label2id, self.__filter_labels
        )

        self.__tokenizer = tokenizer
        self.__max_length = max_length
        self.__dates_window = dates_window

        self.__instances = []

    def __compute_heuristics(self, text: str) -> List[Mapping[str, int | str]]:
        prodigy_spans = []
        if self.__heuristic_dates:
            prodigy_spans = detect_dates(
                text, self.__binarize, window=self.__dates_window
            )
        return prodigy_spans

    def __encode_record(
        self, text: str, eval_entities: List[Mapping[str, int | str]]
    ) -> Iterable[Example]:
        # Filters out entities computed with the heuristics
        training_entities = _discard(eval_entities, self.__filter_labels)
        # Optionally binarizes the entities
        if self.__binarize:
            eval_entities = _binarize(eval_entities)
            training_entities = _binarize(training_entities)
        # Detects entities with heuristics
        heuristic_prodigy_spans = self.__compute_heuristics(text)
        for input_ids, attention_mask, offsets in encode_text(
            text, self.__tokenizer, self.__max_length
        ):
            # List of integer labels used for evaluation
            eval_label_ids = prodigy_to_labels(
                eval_entities, offsets, self.eval_label2id
            )
            # List of training integer labels (used solely for the loss of ML models)
            training_label_ids = prodigy_to_labels(
                training_entities, offsets, self.training_label2id
            )
            # If no heuristics are used, this is a full "O" list
            heuristic_label_ids = prodigy_to_labels(
                heuristic_prodigy_spans, offsets, self.eval_label2id
            )
            # Dictionary of spans used for evaluation
            eval_spans = extract_spans(eval_label_ids, self.__eval_id2label)
            # Tensor of training labels
            labels = torch.tensor(training_label_ids)
            yield input_ids, attention_mask, heuristic_label_ids, eval_spans, labels

    def read_file(self, filepath: Path) -> None:
        # Loads the raw dataset from disk
        logger.info(f"Reading file {filepath}")
        records = list(srsly.read_jsonl(filepath))
        logger.info(f"Read {len(records)} text records from {filepath}")
        if len(self.__discard_labels) > 0:
            records = _discard_tags(records, self.__discard_labels)
            logger.info(f"Tags {self.__discard_labels} discarded")
        # Starts the true encoding
        for record in records:
            self.__instances.extend(
                self.__encode_record(record["text"], record["entities"])
            )
        logger.info(f"{len(self.__instances)} instances obtained from {filepath}")

    def __len__(self) -> int:
        length = len(self.__instances)
        if length == 0:
            raise ValueError("Call `read_file` before accessing this method")
        return length

    def __getitem__(self, index: int) -> Example:
        if len(self.__instances) == 0:
            raise ValueError("Call `read_file` before accessing this method")
        return self.__instances[index]

    @classmethod
    def from_file(
        cls,
        filepath: Path,
        binarize: bool,
        tokenizer: PreTrainedTokenizer,
        heuristic_dates: bool,
        discard_labels: Set[str],
        max_length: int = 512,
        dates_window: Optional[int] = None,
    ) -> "OrdinancesNERDataset":
        dataset = OrdinancesNERDataset(
            binarize,
            tokenizer,
            heuristic_dates,
            discard_labels,
            max_length,
            dates_window,
        )
        dataset.read_file(filepath)
        return dataset

    @classmethod
    def from_files(
        cls,
        filepaths: Iterable[Path],
        binarize: bool,
        tokenizer: PreTrainedTokenizer,
        heuristic_dates: bool,
        discard_labels: Set[str],
        max_length: int = 512,
        dates_window: Optional[int] = None,
    ) -> "OrdinancesNERDataset":
        dataset = OrdinancesNERDataset(
            binarize,
            tokenizer,
            heuristic_dates,
            discard_labels,
            max_length,
            dates_window,
        )
        for filepath in filepaths:
            dataset.read_file(filepath)
        return dataset


class NERDataModule(LightningDataModule):
    def __init__(
        self,
        directory: Path,
        binarize: bool,
        tokenizer: PreTrainedTokenizer,
        heuristic_dates: bool,
        discard_labels: Set[str],
        dates_window: Optional[int] = None,
        batch_size: int = 16,
        num_gpus: int = 1,
        num_workers: int = 8,
        training_filename: str = "training.jsonl",
        validation_filename: str = "validation.jsonl",
        evaluation_filename: str = "evaluation.jsonl",
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.num_workers = num_workers
        # Gets padding token
        pad_token = tokenizer.special_tokens_map["pad_token"]
        self.pad_token_id = tokenizer.get_vocab()[pad_token]

    def _collate_batch(self, batch):
        batch_ = list(zip(*batch))
        input_ids, attention_mask, heuristics, spans, labels = (
            batch_[0],
            batch_[1],
            batch_[2],
            batch_[3],
            batch_[4],
        )

        max_len = max([len(token) for token in input_ids])
        input_ids_tensor = torch.empty(
            size=(len(input_ids), max_len), dtype=torch.long
        ).fill_(self.pad_token_id)
        attention_mask_tensor = torch.zeros(
            size=(len(input_ids), max_len), dtype=torch.bool
        )
        labels_tensor = torch.empty(
            size=(len(input_ids), max_len), dtype=torch.long
        ).fill_(self.eval_label2id["O"])

        for i in range(len(input_ids)):
            tokens_ = input_ids[i]
            seq_len = len(tokens_)

            input_ids_tensor[i, :seq_len] = tokens_
            attention_mask_tensor[i, :seq_len] = attention_mask[i]
            labels_tensor[i, :seq_len] = labels[i]

        return (
            input_ids_tensor,
            attention_mask_tensor,
            heuristics,
            spans,
            labels_tensor,
        )

    @abstractmethod
    def training_warmup_steps(
        self, epochs: int, fraction: float = 0.01
    ) -> Tuple[int, int]:
        pass

    @abstractmethod
    def eval_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class BatchDataModule(NERDataModule):
    def __init__(
        self,
        directory: Path,
        binarize: bool,
        tokenizer: PreTrainedTokenizer,
        heuristic_dates: bool,
        discard_labels: Set[str],
        load_training: bool = True,
        dates_window: int | None = None,
        batch_size: int = 16,
        num_gpus: int = 1,
        num_workers: int = 8,
        training_filename: str = "training.jsonl",
        validation_filename: str = "validation.jsonl",
        evaluation_filename: str = "evaluation.jsonl",
    ) -> None:
        super().__init__(
            directory,
            binarize,
            tokenizer,
            heuristic_dates,
            discard_labels,
            load_training,
            dates_window,
            batch_size,
            num_gpus,
            num_workers,
        )
        # Loads training first
        if load_training:
            self.training = OrdinancesNERDataset.from_file(
                directory / training_filename,
                binarize,
                tokenizer,
                heuristic_dates,
                discard_labels,
                dates_window=dates_window,
            )
        else:
            self.training = None
        # Loads validation and evaluation
        self.validation = OrdinancesNERDataset.from_file(
            directory / validation_filename,
            binarize,
            tokenizer,
            heuristic_dates,
            discard_labels,
            dates_window=dates_window,
        )
        self.evaluation = OrdinancesNERDataset.from_file(
            directory / evaluation_filename,
            binarize,
            tokenizer,
            heuristic_dates,
            discard_labels,
            dates_window=dates_window,
        )
        # Extracts Label to ID mappings
        self.eval_label2id = self.evaluation.eval_label2id
        self.training_label2id = self.evaluation.training_label2id

    def __get_dataloader(self, dataset: OrdinancesNERDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_batch,
            pin_memory=True,
        )

    def training_warmup_steps(
        self, epochs: int, fraction: float = 0.01
    ) -> Tuple[int, int]:
        num_batches = len(self.training) // (self.batch_size * self.num_gpus)
        num_total_steps = epochs * num_batches
        num_warmup_steps = num_total_steps * fraction
        return num_total_steps, num_warmup_steps

    def train_dataloader(self) -> DataLoader:
        if self.training is None:
            raise ValueError(
                "Set `load_training=True` when initializing the OrdinancesNERDataModule."
            )
        return self.__get_dataloader(self.training)

    def val_dataloader(self) -> DataLoader:
        return self.__get_dataloader(self.validation)

    def eval_dataloader(self) -> DataLoader:
        return self.__get_dataloader(self.evaluation)

    def reset(self) -> None:
        pass


class IncrementalDataModule(NERDataModule):
    def __init__(
        self,
        directory: Path,
        binarize: bool,
        tokenizer: PreTrainedTokenizer,
        heuristic_dates: bool,
        discard_labels: Set[str],
        dates_window: int | None = None,
        batch_size: int = 16,
        num_gpus: int = 1,
        num_workers: int = 8,
        training_filename: str = "training.jsonl",
        validation_filename: str = "validation.jsonl",
        evaluation_filename: str = "evaluation.jsonl",
    ) -> None:
        super().__init__(
            directory,
            binarize,
            tokenizer,
            heuristic_dates,
            discard_labels,
            dates_window,
            batch_size,
            num_gpus,
            num_workers,
            training_filename,
            validation_filename,
            evaluation_filename,
        )
        # Loads the unique training/validation dataset
        self.dataset = OrdinancesNERDataset.from_files(
            [directory / training_filename, directory / validation_filename],
            binarize,
            tokenizer,
            heuristic_dates,
            discard_labels,
            dates_window=dates_window,
        )
        # Loads the validation dataset
        self.evaluation = OrdinancesNERDataset.from_file(
            directory / evaluation_filename,
            binarize,
            tokenizer,
            heuristic_dates,
            discard_labels,
            dates_window=dates_window,
        )
        self.eval_label2id = self.evaluation.eval_label2id
        self.training_label2id = self.evaluation.training_label2id
        self.curr = 0

    def training_warmup_steps(
        self, epochs: int, fraction: float = 0.01
    ) -> Tuple[int, int]:
        return 50_000, 0

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset[: self.curr],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_batch,
        )

    def eval_dataloader(self) -> DataLoader:
        return DataLoader(
            self.evaluation,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_batch,
            pin_memory=True,
        )

    def reset(self) -> None:
        self.curr = 0


class WindowDataModule(IncrementalDataModule):
    def __init__(
        self,
        directory: Path,
        binarize: bool,
        tokenizer: PreTrainedTokenizer,
        heuristic_dates: bool,
        discard_labels: Set[str],
        dates_window: int | None = None,
        batch_size: int = 16,
        num_gpus: int = 1,
        num_workers: int = 8,
        training_filename: str = "training.jsonl",
        validation_filename: str = "validation.jsonl",
        evaluation_filename: str = "evaluation.jsonl",
    ) -> None:
        super().__init__(
            directory,
            binarize,
            tokenizer,
            heuristic_dates,
            discard_labels,
            dates_window,
            batch_size,
            num_gpus,
            num_workers,
            training_filename,
            validation_filename,
            evaluation_filename,
        )
        self.curr = 0

    def train_dataloader(self) -> DataLoader:
        # Increments the current counter
        self.curr += self.batch_size
        # Selects the newest examples
        new = self.dataset[self.curr - self.batch_size : self.curr]
        # Rest of the previous data
        rest = self.dataset[: self.curr - self.batch_size]
        # If rest is empty, returns
        if len(rest) == 0:
            dataset = new
        else:
            # Samples at most `2 * batch_size` old records
            old_size = min(2 * self.batch_size, len(rest))
            old = rest[-old_size:]
            dataset = ConcatDataset([old, new])
        # Builds the dataloader
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.batch_size,
            collate_fn=self._collate_batch,
        )


class RandomDataModule(IncrementalDataModule):
    def __init__(
        self,
        directory: Path,
        binarize: bool,
        tokenizer: PreTrainedTokenizer,
        heuristic_dates: bool,
        discard_labels: Set[str],
        dates_window: int | None = None,
        batch_size: int = 16,
        num_gpus: int = 1,
        num_workers: int = 8,
        training_filename: str = "training.jsonl",
        validation_filename: str = "validation.jsonl",
        evaluation_filename: str = "evaluation.jsonl",
    ) -> None:
        super().__init__(
            directory,
            binarize,
            tokenizer,
            heuristic_dates,
            discard_labels,
            dates_window,
            batch_size,
            num_gpus,
            num_workers,
            training_filename,
            validation_filename,
            evaluation_filename,
        )
        # Sets the running index in order to select initially only the first batch
        self.curr = 0

    def train_dataloader(self) -> DataLoader:
        # Increments the current counter
        self.curr += self.batch_size
        # Selects the newest examples
        new = self.dataset[self.curr - self.batch_size : self.curr]
        # Rest of the previous data
        rest = self.dataset[: self.curr - self.batch_size]
        # If rest is empty, returns
        if len(rest) == 0:
            dataset = new
        else:
            # Samples at most `2 * batch_size` old records
            old_size = min(2 * self.batch_size, len(rest))
            discard_size = len(rest) - old_size
            old, _ = random_split(rest, [old_size, discard_size])
            dataset = ConcatDataset([old, new])
        # Builds the dataloader
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.batch_size,
            collate_fn=self._collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset[: self.curr],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_batch,
        )

    def eval_dataloader(self) -> DataLoader:
        return DataLoader(
            self.evaluation,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_batch,
        )


def load_domain_adaptation(
    directory: Path,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    training_filename: str = "training.jsonl",
    validation_filename: str = "validation.jsonl",
    evaluation_filename: str = "evaluation.jsonl",
) -> DatasetDict:
    # Function that encodes the text data
    def __encode_batch(batch: Mapping[str, List]) -> Mapping[str, List]:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_special_tokens_mask=True,
        )

    # Loads the full dataset from disk
    dataset = load_dataset(
        "json",
        data_dir=directory,
        data_files={
            "training": training_filename,
            "validation": validation_filename,
            "evaluation": evaluation_filename,
        },
    )
    # Encodes the batch
    return dataset.map(
        __encode_batch, batched=True, remove_columns=dataset["training"].column_names
    )
