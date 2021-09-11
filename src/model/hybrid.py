from functools import partial

import torch
from torch import nn

from src.model.layers.conv_block import BottleNeck
from src.model.resnet import ResNet
from src.model.vit import build_vit
from src.utils import load_from_zoo


class Hybrid(nn.Module):
    def __init__(self, cnn, vit, fc):
        super(Hybrid, self).__init__()
        self.cnn = cnn
        self.vit = vit
        self.fc = fc

    def predict(self, x):
        x = self.cnn.features(x)
        return self.fc(self.vit(x))

    def forward(self, x):
        return self.predict(x)

    def load_npz(self, npz):
        self.cnn.load_npz(npz)
        self.vit.load_npz(npz)


def get_hybrid(model_name, nclass=1000, pretrained=False, pre_logits=False):
    if 'vit_base' in model_name:
        num_layer, d_model, h, d_ff, N = [3, 4, 9], 768, 12, 3072, 12
    elif 'vit_large' in model_name:
        num_layer, d_model, h, d_ff, N = [3, 4, 6, 3], 1024, 16, 4096, 24

    cnn = ResNet(BottleNeck, num_layer, nclass=nclass, norm_layer=partial(nn.GroupNorm, 32))
    feature_dim, feature_size = get_feature_map_info(cnn, model_name)
    vit = build_vit(patch_size=(1, 1), img_size=feature_size, in_channel=feature_dim, d_model=d_model,
                    h=h, d_ff=d_ff, N=N, nclass=0, pre_logits=pre_logits)
    fc = nn.Linear(d_model, nclass)
    hybrid = Hybrid(cnn=cnn, vit=vit, fc=fc)

    if pretrained:
        load_from_zoo(hybrid, model_name)

    return hybrid


@torch.no_grad()
def get_feature_map_info(cnn, model_name):
    if '224' in model_name:
        img_size = (1, 3, 224, 224)
    elif '384' in model_name:
        img_size = (1, 3, 384, 384)
    _, c, h, w = map(int, cnn.features(torch.rand(img_size)).shape)
    return c, (h, w)