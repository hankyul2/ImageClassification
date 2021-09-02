import torch

from src.vit import get_vit, Embedding, PositionalEncoding


def test_get_vit():
    vit = get_vit(img_size=(224, 224), patch_size=(16, 16))
    assert vit


def test_embedding():
    x = torch.rand((8, 3, 224, 224))
    embed = Embedding()
    out = embed(x)
    assert list(out.shape) == [8, 197, 512]


def test_positional_encoding():
    x = torch.rand((8, 3, 224, 224))
    embed = Embedding()
    pe = PositionalEncoding()
    out = pe(embed(x))
    assert list(out.shape) == [8, 197, 512]
