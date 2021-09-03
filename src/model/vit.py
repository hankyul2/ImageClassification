import math

import torch
from einops import rearrange, repeat
from numpy.testing._private.parameterized import param
from torch import nn
import torch.nn.functional as F

import copy
import re

from src.utils import load_from_zoo


def is_pair(img_size):
    return img_size if isinstance(img_size, tuple) else (img_size, img_size)


def clone(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


class Embedding(nn.Module):
    def __init__(self, d_model=512, img_size=(224, 224), patch_size=(16, 16), in_channel=3, dropout=0.1):
        super(Embedding, self).__init__()
        i_h, i_w = is_pair(img_size)
        p_h, p_w = is_pair(patch_size)
        patch_num = (i_h // p_h) * (i_w // p_w)
        assert i_h % p_h == 0 and i_w % p_w == 0

        self.linear_projection = nn.Conv2d(in_channel, d_model, p_h, p_h)
        self.img2patch = lambda x: rearrange(x, 'b e p q -> b (p q) e')
        self.cls_token = nn.Parameter(torch.rand((1, 1, d_model,)))
        self.pad_cls_token = lambda x: torch.cat([repeat(self.cls_token, '1 1 d -> b 1 d', b=x.size(0)), x], dim=1)
        self.pe = nn.Parameter(torch.rand(1, patch_num + 1, d_model))
        self.add_positional_encoding = lambda x: x + self.pe[:, :x.size(1)]
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear_projection(x)
        x = self.img2patch(x)
        x = self.pad_cls_token(x)
        xx = self.add_positional_encoding(x)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.scale = math.sqrt(self.d_k)

        self.qkv = clone(layer=nn.Linear(d_model, d_model), N=3)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        q, k, v = [rearrange(f(x), 'b s (h k) -> b h s k', k=self.d_k) for f, x in zip(self.qkv, [q, k, v])]
        score = q @ k.transpose(-1, -2) / self.scale
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


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean, std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
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
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def dim_convertor(name, weight):
    weight = torch.from_numpy(weight)
    if 'kernel' in name:
        if weight.dim() == 2:
            weight = rearrange(weight, 'in out -> out in')
        elif weight.dim() == 3:
            if 'out' in name:
                weight = rearrange(weight, 'h k d -> d (h k)')
            else:
                weight = rearrange(weight, 'd h k -> (h k) d')
        elif weight.dim() == 4:
            weight = rearrange(weight, 'h w in_c out_c -> out_c in_c h w')
    elif 'bias' in name:
        if weight.dim() == 2:
            weight = rearrange(weight, 'h k -> (h k)')
    return weight


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

    def load_npz(self, npz):
        name_convertor = [
            ('embed.cls_token', 'cls'), ('embed.linear_projection', 'embedding'),
            ('embed.pe', 'Transformer/posembed_input/pos_embedding'),
            ('s.0.norm', 'LayerNorm_0'), ('s.1.norm', 'LayerNorm_2'), ('a_2', 'scale'), ('b_2', 'bias'),
            ('encoder', 'Transformer'), ('layers.', 'encoderblock_'), ('norm', 'encoder_norm'),
            ('weight', 'kernel'), ('ff', 'MlpBlock_3'), ('w1', 'Dense_0'), ('w2', 'Dense_1'),
            ('attn', 'MultiHeadDotProductAttention_1'), ('qkv.0', 'query'), ('qkv.1', 'key'), ('qkv.2', 'value'), ('\\.', '/')
        ]
        for name, param in self.named_parameters():
            if 'head' in name:
                continue
            for pattern, sub in name_convertor:
                name = re.sub(pattern, sub, name)
            param.data.copy_(dim_convertor(name, npz.get(name)))



def build_vit(d_model=512, h=8, d_ff=2048, N=6, patch_size=(16, 16), img_size=(224, 224),
              nclass=1000, dropout=0.1, in_channel=3):
    c = copy.deepcopy
    embed = Embedding(d_model=d_model, img_size=img_size, patch_size=patch_size, in_channel=in_channel, dropout=dropout)
    attn = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
    ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    su = SublayerConnection(d_model=d_model)
    mlp_head = nn.Linear(d_model, nclass)

    vit = VIT(embed=embed, encoder=Encoder(EncoderLayer(c(attn), c(ff), c(su)), d_model, N), mlp_head=mlp_head)

    for name, param in vit.named_parameters():
        if param.dim() > 2:
            nn.init.kaiming_normal_(param)
        elif param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

    return vit


def get_vit(model_name: str, nclass=1000, pretrained=False):
    '''model_name_form: vit_{base,large}_patch{16,32}_{224,384}'''
    if 'vit_base' in model_name:
        d_model, h, d_ff, N = 768, 12, 3072, 12
    elif 'vit_large' in model_name:
        d_model, h, d_ff, N = 1024, 16, 4096, 24

    if 'patch16' in model_name:
        patch_size = (16, 16)
    elif 'patch32' in model_name:
        patch_size = (32, 32)

    if '224' in model_name:
        img_size = (224, 224)
    elif '384' in model_name:
        img_size = (384, 384)

    vit = build_vit(patch_size=patch_size, img_size=img_size, d_model=d_model, h=h, d_ff=d_ff, N=N, nclass=nclass)

    if pretrained:
        load_from_zoo(vit, model_name)

    return vit
