from itertools import combinations
from sentence_transformers.models import Pooling

import torch
from transformers import AutoModel


class TextEmbedder(torch.nn.Module):
    """https://lightning-flash.readthedocs.io/en/latest/reference/text_embedder.html"""

    def __init__(self, backbone: str):
        super().__init__()
        self.module = AutoModel.from_pretrained(backbone)
        self.pooling = Pooling(self.module.config.hidden_size)

    def forward(self, batch):
        output_states = self.module(**batch)
        output_tokens = output_states.last_hidden_state
        batch.update({"token_embeddings": output_tokens})
        return self.pooling(batch)["sentence_embedding"]


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
