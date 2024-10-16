"""
Loss Functions
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


class TPPLLMMultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss for TPP-LLM

    This loss expects as input a batch of (description, event sequence) embedding pairs.
    For each description, it treats the corresponding event sequence as a positive pair,
    and all other event sequences in the batch as negative examples.
    """

    def __init__(self, scale: float = 20.0, similarity_fct=F.cosine_similarity):
        """
        Initialize the loss function

        :param scale: scaling factor for the similarity scores
        :param similarity_fct: similarity function
        """
        super(TPPLLMMultipleNegativesRankingLoss, self).__init__()
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, desc_embeddings: Tensor, seq_embeddings: Tensor) -> Tensor:
        """
        Forward function to compute the loss

        :param desc_embeddings: batch of description embeddings, (batch_size, embedding_dim)
        :param seq_embeddings: batch of event sequence embeddings, (batch_size, embedding_dim)
        :return: loss value
        """
        # Normalize embeddings
        normalized_desc_embeddings = F.normalize(desc_embeddings, p=2, dim=-1)
        normalized_seq_embeddings = F.normalize(seq_embeddings, p=2, dim=-1)

        # Compute similarities between descriptions and event sequences
        scores = self.similarity_fct(
            normalized_desc_embeddings.unsqueeze(1), normalized_seq_embeddings.unsqueeze(0), dim=-1,
        ) * self.scale  # (batch_size, batch_size)

        # Compute the loss
        labels = torch.arange(scores.size(0), device=scores.device)
        loss = self.cross_entropy_loss(scores, labels)

        return loss
