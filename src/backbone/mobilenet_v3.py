from functools import partial

import torch
from torch import nn

from src.backbone.layers.conv_block import ConvBNAct, MBConvConfig, MBConvSE, mobilenet_v2_init
from src.backbone.mobilenet_v2 import MobileNetV2
from src.backbone.utils import load_from_zoo


class MobileNetV3(MobileNetV2):
    def __init__(self, residual_config, last_channel=1280, dropout=0.2, stochastic_depth=0.0,
                 block=MBConvSE, act_layer=nn.Hardswish, norm_layer=nn.BatchNorm2d):
        super(MobileNetV3, self).__init__(residual_config, dropout, stochastic_depth, block, act_layer, norm_layer)
        in_ch = self.layer_infos[-1].in_ch
        out_ch = in_ch * self.layer_infos[-1].expand_ratio
        self.features[-1] = ConvBNAct(in_ch, out_ch, kernel_size=1, stride=1, norm_layer=self.norm_layer, act=self.act)
        self.classifier = nn.Sequential(
            nn.Linear(out_ch, last_channel),
            act_layer(inplace=True),
        )
        self.out_channels = last_channel

    def forward(self, x):
        return self.dropout(self.classifier(torch.flatten(self.avg_pool(self.features(x)), 1)))


def get_mobilenet_v3(model_name:str, pretrained=True, **kwargs) -> nn.Module:
    """Get mobilenet_v3 large model

    The changes from mobilenet_v3:
        - change input channel to 16 to avoid redundancy
        - apply nn.Hardswish to reduce computational cost
        - apply se unit (from MnasNet)
        - change last stage structure to reduce computational cost
    """

    mbconfig = partial(MBConvConfig, depth_mult=1.0, width_mult=1.0, norm_layer=nn.BatchNorm2d,
                       se_act2=partial(nn.Hardsigmoid, inplace=True), se_reduction_ratio=4, se_reduce_mode='adjust')

    if model_name == 'mobilenet_v3_large':
        residual_config = [
            #    expand k  s  in  out layers act
            mbconfig(1, 3, 1, 16, 16, 1, act=nn.ReLU, use_se=False),
            mbconfig(4, 3, 2, 16, 24, 1, act=nn.ReLU, use_se=False),
            mbconfig(3, 3, 1, 24, 24, 1, act=nn.ReLU, use_se=False),
            mbconfig(3, 5, 2, 24, 40, 1, act=nn.ReLU, use_se=True),
            mbconfig(3, 5, 1, 40, 40, 2, act=nn.ReLU, use_se=True),
            mbconfig(6, 3, 2, 40, 80, 1, act=nn.Hardswish, use_se=False),
            mbconfig(2.5, 3, 1, 80, 80, 1, act=nn.Hardswish, use_se=False),
            mbconfig(2.3, 3, 1, 80, 80, 1, act=nn.Hardswish, use_se=False),
            mbconfig(2.3, 3, 1, 80, 80, 1, act=nn.Hardswish, use_se=False),
            mbconfig(6, 3, 1, 80, 112, 1, act=nn.Hardswish, use_se=True),
            mbconfig(6, 3, 1, 112, 112, 1, act=nn.Hardswish, use_se=True),
            mbconfig(6, 5, 2, 112, 160, 1, act=nn.Hardswish, use_se=True),
            mbconfig(6, 5, 1, 160, 160, 1, act=nn.Hardswish, use_se=True),
            mbconfig(6, 5, 1, 160, 160, 1, act=nn.Hardswish, use_se=True),
        ]
        last_channel = 1280
    elif model_name == 'mobilenet_v3_small':
        residual_config = [
            #    expand k  s  in  out layers act
            mbconfig(1, 3, 2, 16, 16, 1, act=nn.ReLU, use_se=True),
            mbconfig(4.5, 3, 2, 16, 24, 1, act=nn.ReLU, use_se=False),
            mbconfig(3.5, 3, 1, 24, 24, 1, act=nn.ReLU, use_se=False),
            mbconfig(4, 5, 2, 24, 40, 1, act=nn.Hardswish, use_se=True),
            mbconfig(6, 5, 1, 40, 40, 1, act=nn.Hardswish, use_se=True),
            mbconfig(6, 5, 1, 40, 40, 1, act=nn.Hardswish, use_se=True),
            mbconfig(3, 5, 1, 40, 48, 1, act=nn.Hardswish, use_se=True),
            mbconfig(3, 5, 1, 48, 48, 1, act=nn.Hardswish, use_se=True),
            mbconfig(6, 5, 2, 48, 96, 1, act=nn.Hardswish, use_se=True),
            mbconfig(6, 5, 1, 96, 96, 1, act=nn.Hardswish, use_se=True),
            mbconfig(6, 5, 1, 96, 96, 1, act=nn.Hardswish, use_se=True),
        ]
        last_channel = 1024

    model = MobileNetV3(residual_config, last_channel=last_channel, block=MBConvSE, act_layer=nn.Hardswish, norm_layer=nn.BatchNorm2d)

    mobilenet_v2_init(model)

    if pretrained:
        load_from_zoo(model, model_name)

    return model