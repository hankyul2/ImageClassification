"""Squeeze-Excitation Net (SENet, 2018)
This implementation follows timm repo, https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/senet.py
I only add SE ResNet version, which means only on SEBottleNeck exists here.
"""
from torch import nn

from src.model.layers.conv_block import SEBasicBlock, SEBottleNeck
from src.model.resnet import ResNet
from src.utils import load_from_zoo


def get_seresnet(model_name: str, nclass=1000, pretrained=False, dataset=None, **kwargs) -> nn.Module:
    if model_name == 'seresnet18':
        model = ResNet(block=SEBasicBlock, nblock=[2, 2, 2, 2], nclass=nclass)
    elif model_name == 'seresnet34':
        model = ResNet(SEBasicBlock, [3, 4, 6, 3], nclass=nclass)
    elif model_name == 'seresnet50':
        model = ResNet(SEBottleNeck, [3, 4, 6, 3], nclass=nclass)
    elif model_name == 'seresnet101':
        model = ResNet(SEBottleNeck, [3, 4, 23, 3], nclass=nclass)
    elif model_name == 'seresnet152':
        model = ResNet(SEBottleNeck, [3, 8, 36, 3], nclass=nclass)
    elif model_name == 'seresnext50_32x4d':
        model = ResNet(SEBottleNeck, [3, 8, 36, 3], nclass=nclass, groups=32, base_width=4)
    elif model_name == 'wide_seresnet50_2':
        model = ResNet(SEBottleNeck, [3, 8, 36, 3], nclass=nclass, base_width=128)

    if pretrained:
        load_from_zoo(model, model_name)

    if dataset:
        pass

    return model