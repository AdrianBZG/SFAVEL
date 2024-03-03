"""
Module defining a base scorer
"""

import torch
import torch.nn.functional as F

from src.models.scorer.base_scorer import BaseScorer


class L2Scorer(BaseScorer):
    def __init__(self):
        super().__init__()

    def score(self, z_claims, z_triplets):
        query_norm = F.normalize(z_claims, p=2, dim=-1)
        list_norm = F.normalize(z_triplets, p=2, dim=-1)
        if list_norm.size(0) != query_norm.size(0):
            list_norm = list_norm.repeat(query_norm.size(0), 1, 1)
        scores = torch.einsum('ij,ikj->ik', query_norm, list_norm)
        scores = torch.clamp(scores, min=0.0, max=1.0)  # Clip values to stay in [0, 1] range
        return scores
