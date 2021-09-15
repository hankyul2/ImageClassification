"""Squeeze-Excitation Net (SENet, 2018)
This implementation follows timm repo, https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/senet.py
I only add SE ResNet version, which means only on SEBottleNeck exists here.
"""
from collections import OrderedDict

from torch import nn

from src.model.layers.conv_block import SEBasicBlock, SEBottleNeck
from src.model.resnet import ResNet
from src.utils import load_from_zoo


class SeResNet(ResNet):
    def __init__(self, *args, **kwargs):
        super(SeResNet, self).__init__(*args, **kwargs)
        self.layer0 = nn.Sequential(OrderedDict([('conv1', self.conv1), ('bn1', self.bn1), ('relu1', self.relu)]))


def get_seresnet(model_name: str, nclass=1000, pretrained=False, dataset=None, **kwargs) -> nn.Module:
    if model_name == 'seresnet18':
        model = SeResNet(block=SEBasicBlock, nblock=[2, 2, 2, 2], nclass=nclass)
    elif model_name == 'seresnet34':
        model = SeResNet(SEBasicBlock, [3, 4, 6, 3], nclass=nclass)
    elif model_name == 'seresnet50':
        model = SeResNet(SEBottleNeck, [3, 4, 6, 3], nclass=nclass)
    elif model_name == 'seresnet101':
        model = SeResNet(SEBottleNeck, [3, 4, 23, 3], nclass=nclass)
    elif model_name == 'seresnet152':
        model = SeResNet(SEBottleNeck, [3, 8, 36, 3], nclass=nclass)
    elif model_name == 'seresnext50_32x4d':
        model = SeResNet(SEBottleNeck, [3, 8, 36, 3], nclass=nclass, groups=32, base_width=4)
    else:
        raise AssertionError("No model like that in SE ResNet model")

    if pretrained:
        load_from_zoo(model, model_name)

    if dataset:
        pass

    return model