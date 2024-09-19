import pytest

import torch
from src.util import calc_accuracy


class TestMetrics:
    def test_accuracy(self):
        predictions = torch.tensor([[0.3, 0.4, 0.9], [0.4, 0.1, 0.3], [0.9, 0.7, 0.0]])
        label_indices = torch.tensor([0, 0, 0])
        acc = calc_accuracy(predictions, label_indices)
        assert acc == pytest.approx(2 / 3)
