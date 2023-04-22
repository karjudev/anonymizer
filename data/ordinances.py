import os
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Set, Tuple

import torch
from heuristics.date import detect_dates
from log import logger
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
import srsly
from transformers import PreTrainedTokenizer

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


def _binarize_records(
    records: List[Mapping[str, int | str]], label: str = "OMISSIS"
) -> List[Mapping[int, int | str]]:
    for record in records:
        record["entities"] = _binarize(record["entities"], label)
    return records


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


class OrdinancesDataset(Dataset):
    def __init__(
        self,
        binarize: bool,
        tokenizer: PreTrainedTokenizer,
        heuristic_dates: bool,
        discard_labels: Set[str],
        max_length: int = 512,
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

        self.__instances = []

    def __compute_heuristics(self, text: str) -> List[Mapping[str, int | str]]:
        prodigy_spans = []
        if self.__heuristic_dates:
            prodigy_spans = detect_dates(text, self.__binarize)
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
    ) -> "OrdinancesDataset":
        dataset = OrdinancesDataset(
            binarize, tokenizer, heuristic_dates, discard_labels, max_length
        )
        dataset.read_file(filepath)
        return dataset


class OrdinancesDataModule(LightningDataModule):
    def __init__(
        self,
        directory: Path,
        binarize: bool,
        tokenizer: PreTrainedTokenizer,
        heuristic_dates: bool,
        discard_labels: Set[str],
        load_training: bool = True,
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
        # Loads training first
        if load_training:
            self.training = OrdinancesDataset.from_file(
                directory / training_filename,
                binarize,
                tokenizer,
                heuristic_dates,
                discard_labels,
            )
        else:
            self.training = None
        # Loads tuning and validation
        self.validation = OrdinancesDataset.from_file(
            directory / validation_filename,
            binarize,
            tokenizer,
            heuristic_dates,
            discard_labels,
        )
        self.evaluation = OrdinancesDataset.from_file(
            directory / evaluation_filename,
            binarize,
            tokenizer,
            heuristic_dates,
            discard_labels,
        )
        # Extracts Label to ID mappings
        self.eval_label2id = self.evaluation.eval_label2id
        self.training_label2id = self.evaluation.training_label2id

    def num_training_steps(
        self, epochs: int, fraction: float = 0.01
    ) -> Tuple[int, int]:
        num_batches = len(self.training) // (self.batch_size * self.num_gpus)
        num_total_steps = epochs * num_batches
        num_warmup_steps = num_total_steps * fraction
        return num_total_steps, num_warmup_steps

    def __collate_batch(self, batch):
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

    def __get_dataloader(self, dataset: OrdinancesDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.__collate_batch,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        if self.training is None:
            raise ValueError(
                "Set `load_training=True` when initializing the OrdinancesDataModule."
            )
        return self.__get_dataloader(self.training)

    def val_dataloader(self) -> DataLoader:
        return self.__get_dataloader(self.validation)

    def eval_dataloader(self) -> DataLoader:
        return self.__get_dataloader(self.evaluation)
