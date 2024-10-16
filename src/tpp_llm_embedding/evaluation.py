"""
Model Evaluation
"""
import torch
from sentence_transformers import util
from torch import Tensor
from tqdm import tqdm


@torch.no_grad()
def evaluate_similarities(desc_embeddings: Tensor, seq_embeddings: Tensor, top_ks: list = (1, 5, 10)) -> dict:
    """
    Evaluate similarities between event sequences and descriptions

    :param desc_embeddings: event sequence description embeddings
    :param seq_embeddings: event sequence embeddings
    :param top_ks: top K values for retrievals
    :return: recalls at top K values and mean reciprocal rank (MRR)
    """
    assert len(desc_embeddings) == len(seq_embeddings)

    num_samples = len(seq_embeddings)
    corr_retrievals = {top_k: 0 for top_k in top_ks}
    rec_ranks = []

    # Compute cosine similarities for all pairs
    similarities = util.cos_sim(desc_embeddings, seq_embeddings)  # (num_samples, num_samples)

    for i in tqdm(range(num_samples)):
        # Rank the similarities for the i-th description
        top_results = torch.argsort(similarities[i], descending=True).squeeze()

        # Find the rank of the correct sequence
        rank = (top_results == i).nonzero(as_tuple=True)[0].item() + 1
        rec_ranks.append(1.0 / rank)

        # Check if the correct sequence is in the top K for each K
        for top_k in top_ks:
            if i in top_results[:top_k]:
                corr_retrievals[top_k] += 1

    # Calculate recalls at top K and MRR
    total_samples = len(rec_ranks)
    metrics = {f'cosine_recall@{top_k}': corr_retrievals[top_k] / total_samples for top_k in top_ks}
    metrics['cosine_mrr'] = sum(rec_ranks) / total_samples

    return metrics
