import torch

from src.model.resnet import get_resnet


def test_get_resnet():
    x = torch.rand([8, 3, 224, 224])
    resnet18 = get_resnet('resnet18', nclass=50)
    resnet50 = get_resnet('resnet50', nclass=50)
    assert list(resnet18(x).shape) == [8, 50]
    assert list(resnet50(x).shape) == [8, 50]
