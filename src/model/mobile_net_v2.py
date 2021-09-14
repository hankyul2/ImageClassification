import torch
from torch import nn
import torch.nn.functional as F

from src.model.layers.conv_block import InvertedResidualBlock, conv1x1, conv3x3, ConvBNReLU


class MobileNetV2(nn.Module):
    """This implementation follow torchvision works"""
    def __init__(self, nclass, block=InvertedResidualBlock):
        super(MobileNetV2, self).__init__()
        layer_infos = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.norm_layer = nn.BatchNorm2d
        self.act = nn.ReLU6

        self.in_channel = 32
        self.out_channel = 1280

        self.features = nn.Sequential(
            ConvBNReLU(3, self.in_channel, 2, conv3x3, self.norm_layer, self.act),
            *[layer for layer_info in layer_infos for layer in self.make_layers(*layer_info, block)],
            ConvBNReLU(layer_infos[-1][1], self.out_channel, 1, conv1x1, self.norm_layer, self.act)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.out_channel, nclass)
        )

    def make_layers(self, factor, nchannel, nlayer, stride, block):
        layers = []
        for i in range(nlayer):
            layers.append(block(factor, self.in_channel, nchannel, stride=stride,
                                norm_layer=self.norm_layer, act=self.act))
            self.in_channel = nchannel
            stride = 1
        return layers

    def forward(self, x):
        return self.fc(torch.flatten(self.avg_pool(self.features(x)), 1))


def get_mobile_net_v2():
    pass
