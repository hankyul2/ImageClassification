from functools import partial

import torch
from torch import nn

from src.model.layers.conv_block import PreActBottleNeck
from src.model.resnet import ResNet
from src.model.vit import build_vit


class Hybrid(nn.Module):
    def __init__(self, cnn, vit):
        super(Hybrid, self).__init__()
        self.cnn = cnn
        self.vit = vit

    def predict(self, x):
        x = self.cnn.feature(x)
        return self.vit(x)

    def forward(self, x):
        return self.predict(x)


def get_hybrid(model_name, nclass=1000):
    if 'vit_base' in model_name:
        num_layer, d_model, h, d_ff, N = [3, 4, 9], 768, 12, 3072, 12
    elif 'vit_large' in model_name:
        num_layer, d_model, h, d_ff, N = [3, 4, 6, 3], 1024, 16, 4096, 24

    cnn = ResNet(PreActBottleNeck, num_layer, nclass=nclass, norm_layer=partial(nn.GroupNorm, 32))
    c, h, w = get_feature_map_info(cnn)
    print(c, h, w)
    vit = build_vit(patch_size=(1, 1), img_size=(h, w), in_channel=c, d_model=d_model, h=h, d_ff=d_ff, N=N, nclass=nclass)


@torch.no_grad()
def get_feature_map_info(cnn):
    _, c, h, w = map(int, cnn.features(torch.rand((1, 3, 224, 224))).shape)
    return c, h, w