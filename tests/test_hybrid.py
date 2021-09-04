import torch

from src.model.hybrid import get_hybrid


def test_get_hybrid():
    vit = get_hybrid('r50_vit_base_patch16_224')
    assert vit


def test_hybrid_forward():
    x = torch.rand((1, 3, 224, 224))
    vit = get_hybrid('r50_vit_base_patch16_224')
    assert list(vit(x).shape) == [1, 1000]
