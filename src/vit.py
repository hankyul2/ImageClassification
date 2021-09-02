import math

import torch
from einops import rearrange, repeat
from torch import nn
import torch.nn.functional as F

import copy


def is_pair(img_size):
    return img_size if isinstance(img_size, tuple) else (img_size, img_size)


def clone(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


class Embedding(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), d_model=512, dropout=0.1):
        super(Embedding, self).__init__()
        i_h, i_w = is_pair(img_size)
        p_h, p_w = is_pair(patch_size)
        patch_len = (i_h // p_h) * (i_w // p_w)
        patch_size = p_h * p_w * 3
        assert i_h % p_h == 0 and i_w % p_w == 0

        self.img2patch = lambda x: rearrange(x, 'b c (p h) (q w) -> b (p q) (h w c)', h=p_h, w=p_w)
        self.linear_projection = nn.Linear(patch_size, d_model)
        self.cls_token = nn.Parameter(torch.rand((d_model,)))
        self.pad_cls_token = lambda x: torch.cat([repeat(self.cls_token, 'p -> b 1 p', b=x.size(0)), x], dim=1)
        self.pe = nn.Parameter(torch.rand(patch_len + 1, d_model))
        self.add_positional_encoding = lambda x: x + self.pe[:x.size(1)].unsqueeze(0)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.img2patch(x)
        x = self.linear_projection(x)
        x = self.pad_cls_token(x)
        x = self.add_positional_encoding(x)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h

        self.qkv = clone(layer=nn.Linear(d_model, d_model), N=3)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        q, k, v = [rearrange(f(x), 'b s (h k) -> b h s k', k=self.d_k) for f, x in zip(self.qkv, [q, k, v])]
        score = q @ k.transpose(-1, -2) / math.sqrt(self.d_k)
        attn = self.dropout(F.softmax(score, dim=-1))
        v_concat = rearrange(attn @ v, 'b h s k -> b s (h k)')
        return self.out(v_concat)


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
    def __init__(self, encoder, n):
        super(Encoder, self).__init__()
        self.layers = clone(encoder, n)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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


def get_vit(img_size=(224, 224), patch_size=(16, 16), d_model=512, h=8, d_ff=2048, N=3, nclass=1000, dropout=0.1):
    c = copy.deepcopy
    embed = Embedding(img_size=img_size, patch_size=patch_size, d_model=d_model, dropout=dropout)
    attn = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
    ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    su = SublayerConnection(d_model=d_model)
    mlp_head = nn.Linear(d_model, nclass)
    vit = VIT(embed=embed, encoder=Encoder(EncoderLayer(c(attn), c(ff), c(su)), N), mlp_head=mlp_head)

    for name, param in vit.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return vit
