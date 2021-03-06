import torch
import copy

from src.backbone.vit import MultiHeadAttention, FeedForward, Encoder, EncoderLayer, \
    SublayerConnection, build_vit, get_vit, Embedding


def test_build_vit():
    x = torch.rand((8, 3, 224, 224))
    vit = build_vit(img_size=(224, 224), patch_size=(16, 16))
    assert vit
    assert list(vit(x).shape) == [8, 1000]


def test_embedding():
    x = torch.rand((8, 3, 224, 224))
    embed = Embedding()
    out = embed(x)
    assert list(out.shape) == [8, 197, 512]


def test_encoder():
    c = copy.deepcopy
    x = torch.rand((8, 3, 224, 224))
    embed = Embedding()
    out = embed(x)
    attn = MultiHeadAttention(d_model=512, h=8)
    ff = FeedForward(d_model=512, d_ff=2048)
    su = SublayerConnection(d_model=512)
    encoder = Encoder(EncoderLayer(c(attn), c(ff), c(su)), 512, 3)
    assert list(encoder(out).shape) == [8, 197, 512]


def test_get_vit():
    x = torch.rand((8, 3, 224, 224))
    vit = get_vit('vit_base_patch16_224')
    assert vit
    assert list(vit(x).shape) == [8, 1000]

