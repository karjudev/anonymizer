from typing import Callable, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from data.ordinances import Input


class ActiveLearningDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        # Dataset to wrap
        self.__dataset = dataset
        self.label2id = dataset.label2id
        # Mask is 1 iff the record is considered labelled, 0 otherwise
        self.__unlabelled_mask = np.ones(len(self.__dataset), dtype=np.bool_)

    def __len__(self) -> int:
        return len(self.__dataset)

    def __getitem__(self, index: int) -> Tuple[Input, torch.Tensor, int]:
        # Returns the index of the example
        x, y = self.__dataset[index]
        return x, y, index

    def get_labelled(self, batch_size: int, collate_fn: Callable) -> DataLoader:
        labelled_idx = np.where(self.__unlabelled_mask == 0)[0]
        labelled_loader = DataLoader(
            self,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(labelled_idx),
            collate_fn=collate_fn,
        )
        return labelled_loader

    def get_pool(self, batch_size: int, collate_fn: Callable) -> DataLoader:
        # Unlabelled indices
        unlabelled_idx = np.nonzero(self.__unlabelled_mask)[0]
        # Pool i.e. unlabelled training set
        pool_loader = DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=collate_fn,
            sampler=SubsetRandomSampler(unlabelled_idx),
        )
        return pool_loader

    def label(self, index: int) -> None:
        self.__unlabelled_mask[index] = 0
