import os
from pathlib import Path
from typing import Iterable, List, Literal, Mapping, Optional, Set, Tuple

import torch
from log import logger
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
import srsly
from transformers import PreTrainedTokenizer

from data.spans import extract_spans, get_mappings, prodigy_to_labels


# Disable parallel tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Example used for prediction and learning
Example = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
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
    for i, (input_ids, attention_mask, offset_mapping) in enumerate(
        zip(output["input_ids"], output["attention_mask"], output["offset_mapping"])
    ):
        yield input_ids, attention_mask.type(
            torch.bool
        ), offset_mapping, output.word_ids(i)


def _binarize_records(
    records: List[Mapping[str, int | str]], label: str = "OMISSIS"
) -> List[Mapping[int, int | str]]:
    for record in records:
        for entity in record["entities"]:
            entity["label"] = label
    return records


def _discard_tags(
    records: List[Mapping[str, int | str]], tags: Set[str]
) -> List[Mapping[int, int | str]]:
    for record in records:
        filtered_entities = []
        for entity in record["entities"]:
            if entity["label"] not in tags:
                filtered_entities.append(entity)
        record["entities"] = filtered_entities
    return records


class OrdinancesDataset(Dataset):
    def __init__(
        self,
        binarize: bool,
        tokenizer: PreTrainedTokenizer,
        ignore_tags: Set[str] = None,
        label2id: Optional[Mapping[str, int]] = None,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.__binarize = binarize
        self.__ignore_tags = ignore_tags
        self.__tokenizer = tokenizer
        self.__label2id = label2id
        self.__id2label = (
            None
            if label2id is None
            else {idx: label for label, idx in label2id.items()}
        )
        self.__max_length = max_length
        self.__instances = []

    def __compute_token_mask(self, word_ids: List[int]) -> torch.Tensor:
        token_mask = []
        prev_word_idx = None
        for word_idx in word_ids:
            token_mask.append(word_idx is not None and prev_word_idx != word_idx)
            prev_word_idx = word_idx
        return torch.tensor(token_mask, dtype=torch.bool)

    def __encode_record(
        self, text: str, entities: List[Mapping[str, int | str]]
    ) -> Iterable[Example]:
        for input_ids, attention_mask, offsets, word_ids in encode_text(
            text, self.__tokenizer, self.__max_length
        ):
            # True if the token is the first of its word, False otherwise
            token_mask = self.__compute_token_mask(word_ids)
            # List of integer labels
            label_ids = prodigy_to_labels(entities, offsets, self.__label2id)
            # Dictionary of spans
            enc_spans = extract_spans(label_ids, self.__id2label)
            labels = torch.tensor(label_ids)
            yield input_ids, attention_mask, token_mask, enc_spans, labels

    def read_file(self, filepath: Path) -> None:
        # Loads the raw dataset from disk
        logger.info(f"Reading file {filepath}")
        records = list(srsly.read_jsonl(filepath))
        logger.info(f"Read {len(records)} text records from {filepath}")
        if self.__ignore_tags is not None:
            records = _discard_tags(records, self.__ignore_tags)
            logger.info(f"Tags {self.__ignore_tags} discarded")
        if self.__binarize:
            records = _binarize_records(records)
            logger.info("Dataset binarized")
        if self.__label2id is None and self.__id2label is None:
            self.__label2id, self.__id2label = get_mappings(records)
            logger.info(
                f"Mappings correctly computed. We have {len(self.__label2id)} tags to predict"
            )
        # Starts the true encoding
        for record in records:
            self.__instances.extend(
                self.__encode_record(record["text"], record["entities"])
            )
        logger.info(f"{len(self.__instances)} instances obtained from {filepath}")

    def get_target_vocab(self) -> Mapping[str, int]:
        if self.__label2id is None:
            raise ValueError("Call `read_file` before accessing this method")
        return self.__label2id

    def get_target_size(self) -> int:
        vocab = self.get_target_vocab()
        return len(vocab.values())

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
        ignore_tags: Set[str] = None,
        label2id: Optional[Mapping[str, int]] = None,
        max_length: int = 512,
    ) -> "OrdinancesDataset":
        dataset = OrdinancesDataset(
            binarize, tokenizer, ignore_tags, label2id, max_length
        )
        dataset.read_file(filepath)
        return dataset


class OrdinancesDataModule(LightningDataModule):
    def __init__(
        self,
        directory: Path,
        binarize: bool,
        tokenizer: PreTrainedTokenizer,
        ignore_tags: Set[str] = None,
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
        # Gets padding and separation token
        self.pad_token = tokenizer.special_tokens_map["pad_token"]
        self.pad_token_id = tokenizer.get_vocab()[self.pad_token]
        self.sep_token = tokenizer.special_tokens_map["sep_token"]
        # Loads training first
        self.training = OrdinancesDataset.from_file(
            directory / training_filename, binarize, tokenizer, ignore_tags
        )
        # Extracts Label to ID mapping
        self.tag_to_id = self.training.get_target_vocab()
        # Loads tuning and validation
        self.validation = OrdinancesDataset.from_file(
            directory / validation_filename,
            binarize,
            tokenizer,
            ignore_tags,
            self.tag_to_id,
        )
        self.evaluation = OrdinancesDataset.from_file(
            directory / evaluation_filename,
            binarize,
            tokenizer,
            ignore_tags,
            self.tag_to_id,
        )

    def num_training_steps(
        self, epochs: int, fraction: float = 0.01
    ) -> Tuple[int, int]:
        num_batches = len(self.training) // (self.batch_size * self.num_gpus)
        num_total_steps = epochs * num_batches
        num_warmup_steps = num_total_steps * fraction
        return num_total_steps, num_warmup_steps

    def __collate_batch(self, batch):
        batch_ = list(zip(*batch))
        tokens, masks, token_masks, gold_spans, tags = (
            batch_[0],
            batch_[1],
            batch_[2],
            batch_[3],
            batch_[4],
        )

        max_len = max([len(token) for token in tokens])
        token_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(
            self.pad_token_id
        )
        tag_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(
            self.tag_to_id["O"]
        )
        mask_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
        token_masks_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)

        for i in range(len(tokens)):
            tokens_ = tokens[i]
            seq_len = len(tokens_)

            token_tensor[i, :seq_len] = tokens_
            tag_tensor[i, :seq_len] = tags[i]
            mask_tensor[i, :seq_len] = masks[i]
            token_masks_tensor[i, :seq_len] = token_masks[i]

        return token_tensor, tag_tensor, mask_tensor, token_masks_tensor, gold_spans

    def __get_dataloader(self, dataset: OrdinancesDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.__collate_batch,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.__get_dataloader(self.training)

    def val_dataloader(self) -> DataLoader:
        return self.__get_dataloader(self.validation)

    def eval_dataloader(self) -> DataLoader:
        return self.__get_dataloader(self.evaluation)
