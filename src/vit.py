import torch
from einops import rearrange, repeat
from torch import nn


class VIT(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16)):
        super(VIT, self).__init__()


class Embedding(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), d_model=512):
        super(Embedding, self).__init__()
        self.patch_len = patch_size[0] * patch_size[1] * 3
        self.img2patch = lambda x: rearrange(x, 'b c (p h) (q w) -> b (p q) (h w c)', h=patch_size[0], w=patch_size[1])
        self.cls_token = nn.Parameter(torch.rand((self.patch_len,)))
        self.pad_cls_token = lambda x: torch.cat([repeat(self.cls_token, 'p -> b 1 p', b=x.size(0)), x], dim=1)
        self.linear_projection = nn.Linear(self.patch_len, d_model)

    def forward(self, x):
        x = self.img2patch(x)
        x = self.pad_cls_token(x)
        x = self.linear_projection(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, patch_len=197, d_model=512):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.rand(patch_len, d_model))

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


def get_vit(img_size=(224, 224), patch_size=(16, 16)):
    embedding = Embedding(img_size=img_size, patch_size=patch_size)
    vit = VIT(img_size=img_size, patch_size=patch_size)
    return vit
