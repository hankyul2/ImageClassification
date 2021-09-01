from typing import Type, Union

from torch import nn

from src.conv_block import BasicBlock, BottleNeck, conv1x1, resnet_normal_init, resnet_zero_init
from src.utils import load_from_zoo


class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, BottleNeck]], nblock: list, nclass: int = 1000,
                 channels: list = [64, 128, 256, 512], norm_layer: nn.Module = nn.BatchNorm2d, groups=1,
                 base_width=64) -> None:
        super(ResNet, self).__init__()
        self.groups = groups
        self.base_width = base_width
        self.norm_layer = norm_layer
        self.in_channels = channels[0]

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=(7, 7), stride=2, padding=(1, 1), bias=False)
        self.bn1 = self.norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[-1] * block.factor, nclass)

        self.layers = [self.make_layer(block=block, nblock=nblock[i], channels=channels[i]) for i in range(len(nblock))]
        self.register_layer()

    def register_layer(self):
        for i, layer in enumerate(self.layers):
            exec('self.layer{} = {}'.format(i + 1, 'layer'))

    def make_layer(self, block: Type[Union[BasicBlock, BottleNeck]], nblock: int, channels: int) -> nn.Sequential:
        layers = []
        downsample = None
        stride = 1
        if self.in_channels != channels * block.factor:
            stride = 2
            downsample = nn.Sequential(
                conv1x1(self.in_channels, channels * block.factor, stride=stride),
                nn.BatchNorm2d(channels * block.factor)
            )
        for i in range(nblock):
            if i == 1:
                stride = 1
                downsample = None
                self.in_channels = channels * block.factor
            layers.append(block(in_channels=self.in_channels, out_channels=channels,
                                stride=stride, norm_layer=self.norm_layer, downsample=downsample,
                                groups=self.groups, base_width=self.base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for layer in self.layers:
            x = layer(x)
        return self.fc(self.flatten(self.avgpool(x)))


def get_resnet(model_name: str, nclass=1000, zero_init_residual=False, pretrained=False, dataset=None) -> nn.Module:
    if model_name == 'resnet18':
        model = ResNet(block=BasicBlock, nblock=[2, 2, 2, 2], nclass=nclass)
    elif model_name == 'resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], nclass=nclass)
    elif model_name == 'resnet50':
        model = ResNet(BottleNeck, [3, 4, 6, 3], nclass=nclass)
    elif model_name == 'resnet101':
        model = ResNet(BottleNeck, [3, 4, 23, 3], nclass=nclass)
    elif model_name == 'resnet152':
        model = ResNet(BottleNeck, [3, 8, 36, 3], nclass=nclass)
    elif model_name == 'resnext50_32x4d':
        model = ResNet(BottleNeck, [3, 8, 36, 3], nclass=nclass, groups=32, base_width=4)
    elif model_name == 'wide_resnet50_2':
        model = ResNet(BottleNeck, [3, 8, 36, 3], nclass=nclass, base_width=128)

    resnet_normal_init(model)
    resnet_zero_init(model, zero_init_residual)

    if pretrained:
        load_from_zoo(model, model_name, pretrained)

    if dataset:
        pass

    return model


