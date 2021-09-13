from torch import nn
import torch.nn.functional as F

from src.model.layers.conv_block import InvertedResidualBlock, conv1x1, conv3x3


class MobileNetV2(nn.Module):
    def __init__(self, nclass, block=InvertedResidualBlock):
        super(MobileNetV2, self).__init__()
        t = [1, 6, 6, 6, 6, 6, 6]
        c = [16, 24, 32, 64, 96, 160, 320]
        n = [1, 2, 3, 4, 3, 3, 1]
        s = [1, 2, 2, 2, 1, 2, 1]
        self.norm_layer = nn.BatchNorm2d
        self.dropout = 0.1

        self.in_channel = 32
        self.conv1 = conv3x3(3, 32, 2)
        self.layers = nn.ModuleList([self.make_layers(t[i], c[i], n[i], s[i], block) for i in range(7)])
        self.conv2 = conv1x1(320, 1280)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = conv1x1(1280, nclass)

    def make_layers(self, factor, nchannel, nlayer, stride, block):
        layers = []
        downsample = None

        if stride != 0 or self.in_channel != nchannel:
            downsample = nn.Sequential(
                conv1x1(self.in_channel, nchannel, stride),
                self.norm_layer(nchannel)
            )

        for i in range(nlayer):
            layers.append(block(factor, self.in_channel, nchannel, stride=stride,
                                norm_layer=self.norm_layer, downsample=downsample, dropout=self.dropout))
            self.in_channel = nchannel
            stride = 1
            downsample = None
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc(self.avg_pool(self.conv2(x)))


def get_mobile_net_v2():
    pass
