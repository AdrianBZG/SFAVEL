"""
Module defining the loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses


class DistillationLoss(nn.Module):
    def __init__(self):
        super(DistillationLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x_lm, candidates_lm_z):
        # Match shapes
        x_lm = x_lm.unsqueeze(1).expand(candidates_lm_z.shape[0], candidates_lm_z.shape[1], x_lm.shape[1])
        x_lm = x_lm.reshape(-1, x_lm.shape[2])
        candidates_lm_z = candidates_lm_z.reshape(-1, candidates_lm_z.shape[2])

        # Normalize to unit vectors
        x_lm = torch.stack([F.normalize(x, dim=-1) for x in x_lm])
        candidates_lm_z = torch.stack([F.normalize(x, dim=-1) for x in candidates_lm_z])

        logits = x_lm @ candidates_lm_z.transpose(-2, -1)
        labels = torch.ones(len(x_lm), dtype=torch.long, device=x_lm.device)
        loss = self.loss_func(logits, labels)
        return loss


class IntraSampleLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(IntraSampleLoss, self).__init__()
        self.temperature = temperature
        self.loss_func = losses.SupConLoss(temperature=self.temperature)

    def forward(self, x_lm, candidates_z, negatives_z):
        t_loss = 0.0
        batch_size = x_lm.size(0)
        num_positives = (1 + candidates_z.size(1))
        num_negatives = negatives_z.size(1)
        labels = torch.arange(0, num_positives + num_negatives)
        labels[:num_positives] = 0

        for b_id in range(0, batch_size):
            embeddings = torch.cat([x_lm.unsqueeze(1), candidates_z, negatives_z], dim=1)
            embeddings = embeddings[b_id, :, :]
            loss = self.loss_func(embeddings, labels)
            t_loss += loss

        t_loss /= batch_size
        return t_loss


class ScoringLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(ScoringLoss, self).__init__()
        self.loss_func = nn.MarginRankingLoss(margin=margin)

    def forward(self, pos_scores, neg_scores):
        target = torch.ones(max(pos_scores.shape, neg_scores.shape),
                            dtype=torch.long,
                            device=pos_scores.device)

        if pos_scores.size(1) < neg_scores.size(1):
            pos_scores = torch.repeat_interleave(pos_scores, neg_scores.size(1) // pos_scores.size(1), dim=1)
        elif pos_scores.size(1) > neg_scores.size(1):
            neg_scores = torch.repeat_interleave(neg_scores, pos_scores.size(1) // neg_scores.size(1), dim=1)

        return self.loss_func(pos_scores, neg_scores, target)
