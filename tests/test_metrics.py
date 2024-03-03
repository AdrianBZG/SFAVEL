import pytest
import torch


def test_accuracy_all_correct():
    y_pred = torch.tensor([1, 0, 1])
    y_true = torch.tensor([1, 0, 1])
    accuracy = (y_pred.round() == y_true).float().mean()
    assert accuracy == 1.0


def test_accuracy_all_wrong():
    y_pred = torch.tensor([1, 0, 1])
    y_true = torch.tensor([0, 1, 0])
    accuracy = (y_pred.round() == y_true).float().mean()
    assert accuracy == 0.0


def test_accuracy_mixed():
    y_pred = torch.tensor([1, 0, 1])
    y_true = torch.tensor([1, 1, 1])
    accuracy = (y_pred.round() == y_true).float().mean()
    assert accuracy.item() == pytest.approx(0.66666, 0.01)
