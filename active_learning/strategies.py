import torch


def entropy(log_prob: torch.Tensor) -> torch.Tensor:
    # Probability of the labels
    prob = torch.exp(log_prob)
    # Entropy of the sequence
    entropy = -(prob * log_prob).sum(dim=-1).sum(dim=-1)
    return entropy


def cosine_similarity(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
