import pytest
import torch
import torch.nn.functional as F


def test_distillation_loss():
    x_lm = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
    candidates_lm_z = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
    x_lm = torch.stack([F.normalize(x, dim=-1) for x in x_lm])
    candidates_lm_z = torch.stack([F.normalize(x, dim=-1) for x in candidates_lm_z])

    logits = x_lm @ candidates_lm_z.transpose(-2, -1)
    labels = torch.ones(len(x_lm), dtype=torch.long, device=x_lm.device)
    loss = F.cross_entropy(logits, labels, reduction="mean")
    assert loss.item() == pytest.approx(0.693, 0.01)

def test_intra_sample_loss():
    x_lm = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
    candidates_lm_z = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
    all_elements = torch.cat([x_lm, candidates_lm_z], dim=0)

    pos_indices = [(i, j) for j in range(len(x_lm)) for i in range(len(x_lm))]
    pos_indices = torch.tensor(pos_indices)

    diag_indices = torch.arange(all_elements.size(0)).reshape(all_elements.size(0), 1).expand(-1, 2)

    pos_indices = torch.cat([pos_indices, diag_indices], dim=0)

    target = torch.zeros(all_elements.size(0), all_elements.size(0), device=all_elements.device)
    target[pos_indices[:, 0], pos_indices[:, 1]] = 1.0

    cs = F.cosine_similarity(all_elements[None, :, :], all_elements[:, None, :], dim=-1)
    cs[torch.eye(cs.size(0)).bool()] = float("inf")

    loss = F.binary_cross_entropy((cs / 0.01).sigmoid(), target, reduction="none")

    target_pos = target.bool()
    target_neg = ~target_pos

    loss_pos = torch.zeros(cs.size(0), cs.size(0), device=cs.device).masked_scatter(target_pos, loss[target_pos])
    loss_neg = torch.zeros(cs.size(0), cs.size(0), device=cs.device).masked_scatter(target_neg, loss[target_neg])
    loss_pos = loss_pos.sum(dim=1)
    loss_neg = loss_neg.sum(dim=1)
    num_pos = target.sum(dim=1)
    num_neg = cs.size(0) - num_pos

    loss = ((loss_pos / num_pos) + (loss_neg / num_neg)).mean()
    assert loss.item() == pytest.approx(100.0, 0.01)

def test_scoring_loss():
    pos_scores = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    neg_scores = torch.tensor([5, 6, 7, 8], dtype=torch.float)
    loss_func = torch.nn.MarginRankingLoss(margin=1.0)

    target = torch.ones(pos_scores.shape, dtype=torch.long, device=pos_scores.device)
    loss = loss_func(pos_scores, neg_scores, target)
    assert loss.item() == pytest.approx(5.0, 0.01)
