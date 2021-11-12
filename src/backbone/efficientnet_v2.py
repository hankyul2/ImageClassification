from functools import partial

from torch import nn

from src.backbone.efficientnet import EfficientNet
from src.backbone.layers.conv_block import MBConvConfig, MBConvSE, mobilenet_v2_init
from src.backbone.utils import load_from_zoo


class EfficientNetV2(EfficientNet):
    pass


def get_efficientnet_v2(model_name, pretrained, **kwargs):
    # Todo: Change this to residual config
    # reference(pytorch): https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
    # reference(official): https://github.com/google/automl/blob/master/efficientnetv2/effnetv2_configs.py

    if model_name == 'efficientnet_v2_s':
        depth_mult, width_mult, dropout = 1.0, 1.0, 0.2
    elif model_name == 'efficientnet_v2_m':
        depth_mult, width_mult, dropout = 1.1, 1.0, 0.2
    elif model_name == 'efficientnet_v2_l':
        depth_mult, width_mult, dropout = 1.2, 1.1, 0.3
    elif model_name == 'efficientnet_v2_xl':
        depth_mult, width_mult, dropout = 1.4, 1.2, 0.3


    mbconfig = partial(MBConvConfig, depth_mult=depth_mult, width_mult=width_mult, act=nn.SiLU, norm_layer=nn.BatchNorm2d,
                       se_act1=partial(nn.SiLU, inplace=True), se_reduction_ratio=4, se_reduce_mode='base')

    residual_config = [
        #    expand k  s  in  out layers  se
        mbconfig(1, 3, 1, 24, 24, 2, use_se=False, fused=True),
        mbconfig(4, 3, 2, 24, 48, 4, use_se=False, fused=True),
        mbconfig(4, 3, 2, 48, 64, 4, use_se=False, fused=True),
        mbconfig(4, 3, 2, 64, 128, 6, use_se=True),
        mbconfig(6, 3, 1, 160, 160, 9, use_se=True),
        mbconfig(6, 5, 2, 160, 256, 15, use_se=True),
    ]

    model = EfficientNetV2(residual_config, dropout=dropout, stochastic_depth=0.2, block=MBConvSE, act_layer=nn.SiLU)
    mobilenet_v2_init(model)

    if pretrained:
        load_from_zoo(model, model_name)

    return model

