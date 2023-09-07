from abc import abstractmethod
import os
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Tuple

import torch
from log import logger
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
import srsly
from transformers import PreTrainedTokenizer
from datasets import DatasetDict, load_dataset

from data.spans import extract_spans, get_label2id, prodigy_to_labels


# Disable parallel tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


Input = Tuple[torch.Tensor, torch.Tensor]
Example = Tuple[Input, torch.Tensor]


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


def get_collate_function(
    tokenizer: PreTrainedTokenizer, label2id: Mapping[str, int], active_learning: bool
) -> Callable:
    # Gets padding token
    pad_token = tokenizer.special_tokens_map["pad_token"]
    pad_token_id = tokenizer.get_vocab()[pad_token]
    # Gets numerical index of the "O" label
    out_idx = label2id["O"]

    def collate_fn(batch):
        input_ids, attention_mask, labels, indices = [], [], [], []
        for sample in batch:
            if active_learning:
                (ids, mask), labs, idx = sample
            else:
                (ids, mask), labs = sample
            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(labs)
            if active_learning:
                indices.append(idx)

        max_len = max([len(token) for token in input_ids])
        input_ids_tensor = torch.empty(
            size=(len(input_ids), max_len), dtype=torch.long
        ).fill_(pad_token_id)
        attention_mask_tensor = torch.zeros(
            size=(len(input_ids), max_len), dtype=torch.bool
        )
        labels_tensor = torch.empty(
            size=(len(input_ids), max_len), dtype=torch.long
        ).fill_(out_idx)
        if active_learning:
            indices_tensor = torch.empty(
                size=(len(input_ids), 1), dtype=torch.long
            ).fill_(-1)

        for i in range(len(input_ids)):
            tokens_ = input_ids[i]
            seq_len = len(tokens_)

            input_ids_tensor[i, :seq_len] = tokens_
            attention_mask_tensor[i, :seq_len] = attention_mask[i]
            labels_tensor[i, :seq_len] = labels[i]
            if active_learning:
                indices_tensor[i, 0] = indices[i]
        if active_learning:
            return (
                (input_ids_tensor, attention_mask_tensor),
                labels_tensor,
                indices_tensor,
            )
        return ((input_ids_tensor, attention_mask_tensor), labels_tensor)

    return collate_fn


def training_warmup_steps(
    dataset: Dataset,
    batch_size: int,
    num_gpus: int,
    epochs: int,
    fraction: float = 0.01,
) -> Tuple[int, int]:
    num_batches = len(dataset) // (batch_size * num_gpus)
    num_total_steps = epochs * num_batches
    num_warmup_steps = num_total_steps * fraction
    return num_total_steps, num_warmup_steps


def _binarize(
    entities: Mapping[str, int | str], label: str = "OMISSIS"
) -> Mapping[str, int | str]:
    for entity in entities:
        entity["label"] = label
    return entities


class OrdinancesNERDataset(Dataset):
    def __init__(
        self, binarize: bool, tokenizer: PreTrainedTokenizer, max_length: int = 512
    ) -> None:
        super().__init__()
        self.__binarize = binarize
        self.label2id = get_label2id(binarize)

        self.__tokenizer = tokenizer
        self.__max_length = max_length

        self.__instances = []

    def __encode_record(
        self, text: str, entities: List[Mapping[str, int | str]]
    ) -> Iterable[Example]:
        # Optionally binarizes the entities
        if self.__binarize:
            entities = _binarize(entities)
        # Detects entities with heuristics
        for input_ids, attention_mask, offsets in encode_text(
            text, self.__tokenizer, self.__max_length
        ):
            # List of training integer labels (used solely for the loss of ML models)
            label_ids = prodigy_to_labels(entities, offsets, self.label2id)
            # Tensor of training labels
            labels = torch.tensor(label_ids)
            yield (input_ids, attention_mask), labels

    def read_file(self, filepath: Path) -> None:
        # Loads the raw dataset from disk
        logger.info(f"Reading file {filepath}")
        records = list(srsly.read_jsonl(filepath))
        logger.info(f"Read {len(records)} text records from {filepath}")
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
        max_length: int = 512,
    ) -> "OrdinancesNERDataset":
        dataset = OrdinancesNERDataset(binarize, tokenizer, max_length)
        dataset.read_file(filepath)
        return dataset

    @classmethod
    def from_files(
        cls,
        filepaths: Iterable[Path],
        binarize: bool,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ) -> "OrdinancesNERDataset":
        dataset = OrdinancesNERDataset(binarize, tokenizer, max_length)
        for filepath in filepaths:
            dataset.read_file(filepath)
        return dataset


class NERDataModule(LightningDataModule):
    def __init__(
        self,
        directory: Path,
        binarize: bool,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 16,
        num_gpus: int = 1,
        num_workers: int = 8,
        training_filename: str = "training.jsonl",
        validation_filename: str = "validation.jsonl",
        evaluation_filename: str = "evaluation.jsonl",
        test_filename: str = "test.jsonl",
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.num_workers = num_workers
        # Gets padding token
        pad_token = tokenizer.special_tokens_map["pad_token"]
        self.pad_token_id = tokenizer.get_vocab()[pad_token]

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
        load_training: bool = True,
        batch_size: int = 16,
        num_gpus: int = 1,
        num_workers: int = 8,
        training_filename: str = "training.jsonl",
        validation_filename: str = "validation.jsonl",
        evaluation_filename: str = "evaluation.jsonl",
        test_filename: str = "test.jsonl",
    ) -> None:
        super().__init__(
            directory, binarize, tokenizer, batch_size, num_gpus, num_workers
        )
        # Loads training first
        if load_training:
            self.training = OrdinancesNERDataset.from_file(
                directory / training_filename, binarize, tokenizer
            )
        else:
            self.training = None
        # Loads validation and evaluation
        self.validation = OrdinancesNERDataset.from_file(
            directory / validation_filename, binarize, tokenizer
        )
        self.evaluation = OrdinancesNERDataset.from_file(
            directory / evaluation_filename, binarize, tokenizer
        )
        # Loads test set
        self.test = OrdinancesNERDataset.from_file(
            directory / test_filename, binarize, tokenizer
        )
        # Extracts Label to ID mappings
        self.label2id = self.evaluation.label2id
        self.__collate_batch = get_collate_function(
            tokenizer, self.label2id, active_learning=False
        )

    def __get_dataloader(self, dataset: OrdinancesNERDataset) -> DataLoader:
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
                "Set `load_training=True` when initializing the OrdinancesNERDataModule."
            )
        return self.__get_dataloader(self.training)

    def val_dataloader(self) -> DataLoader:
        return self.__get_dataloader(self.validation)

    def eval_dataloader(self) -> DataLoader:
        return self.__get_dataloader(self.evaluation)

    def test_dataloader(self) -> DataLoader:
        return self.__get_dataloader(self.test)

    def reset(self) -> None:
        pass


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
