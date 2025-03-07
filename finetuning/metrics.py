import torch
import numpy as np
from typing import Dict, List
    
def compute_metrics(image_features, text_features, classes, k_values=[1, 5, 10]):
    """
    Compute retrieval metrics for Image-to-Text (I2T) and Text-to-Image (T2I).

    Args:
        image_features (torch.Tensor): Normalized image features (N x D).
        text_features (torch.Tensor): Normalized text features (N x D).
        classes (torch.Tensor): Class labels for each image-text pair.
        k_values (list): List of K values for Recall@K.

    Returns:
        dict: Dictionary containing metrics for I2T and T2I.
    """
    similarity_matrix = image_features @ text_features.T

    # Labels (diagonal = correct matches)
    labels = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)

    i2t_metrics = _compute_single_direction_metrics(similarity_matrix, labels, k_values)

    t2i_metrics = _compute_single_direction_metrics(similarity_matrix.T, labels, k_values)

    return {
        "I2T": {**i2t_metrics},
        "T2I": {**t2i_metrics}
    }

def _compute_single_direction_metrics(similarity_matrix, labels, k_values):
    """
    Compute retrieval metrics for a single direction (I2T or T2I).

    Args:
        similarity_matrix (torch.Tensor): Similarity matrix (N x N or N x M).
        labels (torch.Tensor): Ground truth indices.
        k_values (list): List of K values for Recall@K.

    Returns:
        dict: Dictionary with Recall@K, MRR, and Median Rank.
    """
    # Sort indices based on similarity scores (descending order)
    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    # Find the rank of the correct label
    ranks = (sorted_indices == labels.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1

    recalls = {f"Recall@{k}": torch.mean((ranks <= k).float()).item() for k in k_values}
    mrr = torch.mean(1.0 / ranks.float()).item()
    median_rank = torch.median(ranks.float()).item()

    metrics = {
        **recalls,
        "MRR": mrr,
        "Median Rank": median_rank
    }
    return metrics