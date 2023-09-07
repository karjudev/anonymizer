from typing import Callable, Sequence, Tuple
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from lightning.pytorch import LightningModule

from active_learning.dataset import ActiveLearningDataset
from active_learning.strategies import cosine_similarity, entropy


QueryStrategy = Callable[
    [
        LightningModule,
        DataLoader,
        ActiveLearningDataset,
        int,
        int | None,
        Callable | None,
    ],
    Sequence[int],
]


def query_the_oracle(
    model: LightningModule,
    dataset: ActiveLearningDataset,
    collate_fn: Callable,
    query_size: int,
    query_strategy: QueryStrategy,
    batch_size: int,
) -> Sequence[int]:
    # Pool i.e. unlabelled training set
    pool_loader = dataset.get_pool(batch_size=batch_size, collate_fn=collate_fn)
    # Deploys the query strategy to select the indices
    sample_idx = query_strategy(
        model=model,
        pool_loader=pool_loader,
        dataset=dataset,
        query_size=query_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    return sample_idx


def random_query(
    model: LightningModule,
    pool_loader: DataLoader,
    dataset: ActiveLearningDataset,
    query_size: int,
    batch_size: int | None = None,
    collate_fn: Callable | None = None,
) -> Sequence[int]:
    # Returns the first K indices because they are already randomized
    sample_idx = []
    for _, _, idx in pool_loader:
        sample_idx.extend(idx.tolist())
        if len(sample_idx) >= query_size:
            break
    return sample_idx[:query_size]


def entropy_query(
    model: LightningModule,
    pool_loader: DataLoader,
    dataset: ActiveLearningDataset,
    query_size: int,
    batch_size: int | None = None,
    collate_fn: Callable | None = None,
) -> Sequence[int]:
    entropy_values = []
    indices = []
    with torch.no_grad():
        for x, _, idx in pool_loader:
            log_prob = model(x)
            batch_entropy = entropy(log_prob)
            entropy_values.append(batch_entropy)
            indices.append(idx)
    entropy_tensor = torch.cat(entropy_values)
    indices = torch.cat(indices)
    # Indices of the top K entropy values
    k = min(query_size, entropy_tensor.size(-1))
    _, ind = torch.topk(entropy_tensor, k=k)
    # Indices in the dataset of the top k
    return indices[ind].tolist()


def __document_embeddings(
    model: LightningModule, loader: DataLoader
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        embedding_values = []
        index_values = []
        for x, _, idx in loader:
            input_ids, attention_mask = x
            batch_embedding = model.embed(
                (input_ids.to("cuda"), attention_mask.to("cuda"))
            )
            embedding_values.append(batch_embedding)
            index_values.append(idx)
        # If the loader is empty, returns None
        if len(embedding_values) == 0:
            return None, None
        return torch.cat(embedding_values), torch.cat(index_values)


def distance_query(
    model: LightningModule,
    pool_loader: DataLoader,
    dataset: ActiveLearningDataset,
    query_size: int,
    batch_size: int | None = None,
    collate_fn: Callable | None = None,
) -> Sequence[int]:
    model = model.to("cuda")
    # Embeddings of the labelled set
    labelled_embeddings, _ = __document_embeddings(
        model, dataset.get_labelled(batch_size, collate_fn)
    )
    # If there are no labelled examples, resorts to random sampling
    if labelled_embeddings is None:
        return random_query(
            model, pool_loader, dataset, query_size, batch_size, collate_fn
        )
    # Embeddings of the pool
    pool_embeddings, pool_indices = __document_embeddings(model, pool_loader)
    with torch.no_grad():
        # Cosine distance between embeddings
        distance = 1.0 - cosine_similarity(pool_embeddings, labelled_embeddings)
        # For each pool example, the maximum distance to a labelled example
        distance, _ = distance.max(dim=-1)
    # Top K indices by distance
    k = min(query_size, distance.size(-1))
    _, ind = torch.topk(distance, k=k)
    return pool_indices[ind.to("cpu")]
