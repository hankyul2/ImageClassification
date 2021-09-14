import torch

from src.model.senet import get_seresnet


def test_get_seresnet():
    x = torch.rand([4, 3, 224, 224])
    model = get_seresnet('seresnet50', nclass=100)
    assert list(model(x).shape) == [4, 100]

