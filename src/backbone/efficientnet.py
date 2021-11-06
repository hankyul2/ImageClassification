import math
from functools import partial

from torch import nn

from src.backbone.utils import load_from_zoo


class MBConvConfig:
    def __init__(self, expand_ratio, kernel, stride, in_ch, out_ch, layers, depth_mult, width_mult):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_ch = self.adjust_channels(in_ch, width_mult)
        self.out_ch = self.adjust_channels(out_ch, width_mult)
        self.num_layers = self.adjust_depth(layers, depth_mult)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += "expand_ratio={expand_ratio}"
        s += ", kernel={kernel}"
        s += ", stride={stride}"
        s += ", input_channels={in_ch}"
        s += ", out_channels={out_ch}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)

    @staticmethod
    def adjust_depth(layers, factor):
        return int(math.ceil(layers * factor))

    @staticmethod
    def adjust_channels(channel, factor):
        new_channel = channel * factor
        divisible_channel = max(8, (int(new_channel + 4) // 8) * 8)
        divisible_channel += 8 if divisible_channel < 0.9 * new_channel else 0
        return divisible_channel


class EfficientNet(nn.Module):
    def __init__(self, config, dropout=0.1):
        super(EfficientNet, self).__init__()


def get_efficientnet(model_name, pretrained, **kwargs):
    if model_name == 'efficientnet_b0':
        depth_mult, width_mult, dropout = 1.0, 1.0, 0.2
    elif model_name == 'efficientnet_b1':
        depth_mult, width_mult, dropout = 1.1, 1.0, 0.2
    elif model_name == 'efficientnet_b2':
        depth_mult, width_mult, dropout = 1.2, 1.1, 0.3
    elif model_name == 'efficientnet_b3':
        depth_mult, width_mult, dropout = 1.4, 1.2, 0.3
    elif model_name == 'efficientnet_b4':
        depth_mult, width_mult, dropout = 1.8, 1.4, 0.4
    elif model_name == 'efficientnet_b5':
        depth_mult, width_mult, dropout = 2.2, 1.6, 0.4
    elif model_name == 'efficientnet_b6':
        depth_mult, width_mult, dropout = 2.6, 1.8, 0.5
    elif model_name == 'efficientnet_b7':
        depth_mult, width_mult, dropout = 3.1, 2.0, 0.5

    norm_layer = None if int(model_name[-1]) < 5 else partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

    mbconfig = partial(MBConvConfig, depth_mult=depth_mult, width_mult=width_mult)
    residual_config = [
        mbconfig(1, 3, 1, 16, 24, 1),
        mbconfig(6, 3, 2, 24, 40, 2),
        mbconfig(6, 5, 2, 40, 80, 2),
        mbconfig(6, 3, 2, 80, 112, 3),
        mbconfig(6, 5, 1, 112, 192, 3),
        mbconfig(6, 5, 2, 192, 320, 4),
        mbconfig(6, 3, 1, 320, 1280, 1),
    ]

    model = EfficientNet(residual_config, dropout=dropout, norm_layer=norm_layer)

    if pretrained:
        load_from_zoo(model, model_name)

    return model