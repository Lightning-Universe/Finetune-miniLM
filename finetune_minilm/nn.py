from itertools import combinations

import torch


def cosine_embedding_loss(
    embedding1: torch.Tensor, embedding2: torch.Tensor, label1: torch.Tensor, label2: torch.Tensor
) -> torch.Tensor:
    # label values must be 1 or -1
    match = torch.tensor(1 if label1 == label2 else -1, device=label1.device)
    loss = torch.nn.functional.cosine_embedding_loss(embedding1, embedding2, match, reduction="none")
    return loss


def pairwise_cosine_embedding_loss(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    batch_size = len(embeddings)
    losses = []
    for a, b in combinations(range(batch_size), r=2):
        # compute loss by pairs
        embedding1, embedding2 = embeddings[a], embeddings[b]
        label1, label2 = labels[a], labels[b]
        loss = cosine_embedding_loss(embedding1, embedding2, label1, label2)
        losses.append(loss)
    loss = torch.mean(torch.stack(losses))
    return loss
