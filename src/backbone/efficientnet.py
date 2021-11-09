import copy
from functools import partial

from torch import nn

from src.backbone.layers.conv_block import MBConvConfig, mobilenet_v2_init, MBConvSE
from src.backbone.mobile_net_v2 import MobileNetV2
from src.backbone.utils import load_from_zoo


class EfficientNet(MobileNetV2):
    def make_stages(self, layer_infos, block):
        return [nn.Sequential(*self.make_layers(copy.copy(layer_info), block)) for layer_info in layer_infos]


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

    norm_layer = nn.BatchNorm2d if int(model_name[-1]) < 5 else partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

    mbconfig = partial(MBConvConfig, depth_mult=depth_mult, width_mult=width_mult,
                       use_se=True, se_act1=partial(nn.SiLU, inplace=True), se_reduction_ratio=4, se_divide=True)

    residual_config = [
        #    expand k  s  in  out layers
        mbconfig(1, 3, 1, 32, 16, 1),
        mbconfig(6, 3, 2, 16, 24, 2),
        mbconfig(6, 5, 2, 24, 40, 2),
        mbconfig(6, 3, 2, 40, 80, 3),
        mbconfig(6, 5, 1, 80, 112, 3),
        mbconfig(6, 5, 2, 112, 192, 4),
        mbconfig(6, 3, 1, 192, 320, 1),
    ]

    model = EfficientNet(residual_config, dropout=dropout, stochastic_depth=0.2, block=MBConvSE, act_layer=nn.SiLU, norm_layer=norm_layer)
    mobilenet_v2_init(model)

    if pretrained:
        load_from_zoo(model, model_name)

    return model