import math

from einops import rearrange
from torch import nn
import torch.nn.functional as F

import copy

from src.model.layers.embed import ConvLinearProjection, TokenLayer, PositionalEncoding, get_patch_num_and_dim
from src.model.model_utils import clone


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.scale = math.sqrt(self.d_k)

        self.qkv = clone(layer=nn.Linear(d_model, d_model), N=3)
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        q, k, v = [rearrange(f(x), 'b s (h k) -> b h s k', k=self.d_k) for f, x in zip(self.qkv, [q, k, v])]
        score = q @ k.transpose(-1, -2) / self.scale
        attn = self.dropout(F.softmax(score, dim=-1))
        v_concat = rearrange(attn @ v, 'b h s k -> b s (h k)')
        return self.proj(v_concat)


class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, layer):
        return x + self.dropout(layer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, attn, ff, su):
        super(EncoderLayer, self).__init__()
        self.s = clone(copy.deepcopy(su), 2)
        self.attn = attn
        self.ff = ff

    def forward(self, x):
        x = self.s[0](x, lambda x: self.attn(x, x, x))
        return self.s[1](x, self.ff)


class Encoder(nn.Module):
    def __init__(self, encoder, d_model, n):
        super(Encoder, self).__init__()
        self.layers = clone(encoder, n)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class VIT(nn.Module):
    def __init__(self, embed, encoder, mlp_head):
        super(VIT, self).__init__()
        self.embed = embed
        self.encoder = encoder
        self.mlp_head = mlp_head

    def forward(self, x):
        x = self.encode(x)
        return self.mlp_head(x[:, 0])

    def predict(self, x):
        x = self.encode(x)
        return self.mlp_head(x[:, 0])

    def encode(self, x):
        return self.encoder(self.embed(x))


def build_vit(img_size=(224, 224), patch_size=(16, 16), d_model=512, h=8, d_ff=2048, N=6, nclass=1000, dropout=0.1, in_channel=3):
    c = copy.deepcopy
    patch_num, patch_dim = get_patch_num_and_dim(img_size, patch_size, in_channel)
    linear_projection = ConvLinearProjection(d_model=d_model, patch_size=patch_size, in_channel=in_channel)
    token_layer = TokenLayer(d_model=d_model)
    positional_encoding = PositionalEncoding(patch_num, d_model, dropout)
    attn = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
    ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    su = SublayerConnection(d_model=d_model)
    mlp_head = nn.Linear(d_model, nclass)

    vit = VIT(embed=nn.Sequential(linear_projection, token_layer, positional_encoding),
              encoder=Encoder(EncoderLayer(c(attn), c(ff), c(su)), d_model, N),
              mlp_head=mlp_head)

    for name, param in vit.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return vit

def get_vit(model_name, nclass=1000):
    if model_name == 'vit_base':
        vit = build_vit(d_model=512, h=8, N=6, nclass=nclass)
    elif model_name == 'vit_large':
        vit = build_vit(d_model=512, h=8, N=6, nclass=nclass)
    elif model_name == 'vit_huge':
        vit = build_vit(d_model=512, h=8, N=6, nclass=nclass)

    return vit