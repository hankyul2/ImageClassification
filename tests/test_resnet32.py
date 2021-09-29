import torch

from src.backbone.resnet32 import get_resnet32


def test_get_resnet32():
    x = torch.rand([8, 3, 32, 32])
    resnet32_20 = get_resnet32('resnet32_20', nclass=20)
    assert list(resnet32_20(x).shape) == [8, 20]

