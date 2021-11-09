import copy
from functools import partial

import torch
from torch import nn

from src.backbone.layers.conv_block import ConvBNAct, mobilenet_v2_init, MBConvConfig, MBConv
from src.backbone.utils import load_from_zoo


class MobileNetV2(nn.Module):
    """This implementation follow torchvision works"""
    def __init__(self, layer_infos, dropout=0.2, stochastic_depth=0.0, block=MBConv, act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super(MobileNetV2, self).__init__()
        self.layer_infos = layer_infos
        self.norm_layer = norm_layer
        self.act = act_layer

        self.in_channel = layer_infos[0].in_ch
        self.last_channels = layer_infos[-1].out_ch
        self.out_channels = self.last_channels * 4

        self.cur_block = 0
        self.num_block = sum(stage.num_layers for stage in layer_infos)
        self.stochastic_depth = stochastic_depth

        self.features = nn.Sequential(
            ConvBNAct(3, self.in_channel, kernel_size=3, stride=2, norm_layer=self.norm_layer, act=self.act),
            *self.make_stages(layer_infos, block),
            ConvBNAct(self.last_channels, self.out_channels, kernel_size=1, stride=1, norm_layer=self.norm_layer, act=self.act)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)

    def make_stages(self, layer_infos, block):
        return [layer for layer_info in layer_infos for layer in self.make_layers(copy.copy(layer_info), block)]

    def make_layers(self, layer_info, block):
        layers = []
        for i in range(layer_info.num_layers):
            layers.append(block(layer_info, sd_prob=self.get_sd_prob()))
            layer_info.in_ch = layer_info.out_ch
            layer_info.stride = 1
        return layers

    def get_sd_prob(self):
        sd_prob = self.stochastic_depth * (self.cur_block / self.num_block)
        self.cur_block += 1
        return sd_prob

    def forward(self, x):
        return self.dropout(torch.flatten(self.avg_pool(self.features(x)), 1))


def get_mobilenet_v2(model_name:str, pretrained=True, **kwargs) -> nn.Module:
    """Get mobilenet_v2 only support 1 model"""
    mbconfig = partial(MBConvConfig, depth_mult=1.0, width_mult=1.0, act=nn.ReLU6, norm_layer=nn.BatchNorm2d)

    residual_config = [
        #    expand k  s  in  out layers
        mbconfig(1, 3, 1, 32, 16, 1),
        mbconfig(6, 3, 2, 16, 24, 2),
        mbconfig(6, 3, 2, 24, 32, 3),
        mbconfig(6, 3, 2, 32, 64, 4),
        mbconfig(6, 3, 1, 64, 96, 3),
        mbconfig(6, 3, 2, 96, 160, 3),
        mbconfig(6, 3, 1, 160, 320, 1),
    ]

    model = MobileNetV2(residual_config)

    mobilenet_v2_init(model)

    if pretrained:
        load_from_zoo(model, model_name)

    return model

