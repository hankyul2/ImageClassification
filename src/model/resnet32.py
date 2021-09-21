from typing import Type, Union

from pytorch_lightning import LightningModule
from torch import nn

from src.model.layers.conv_block import BasicBlock, BottleNeck, conv1x1, resnet_normal_init, resnet_zero_init, \
    PreActBasicBlock, PreActBottleNeck, InvertedResidualBlock, SEBasicBlock, SEBottleNeck


class ResNet32(LightningModule):
    def __init__(self,
                 block: Type[Union[BasicBlock, BottleNeck,
                                   PreActBasicBlock, PreActBottleNeck,
                                   InvertedResidualBlock,
                                   SEBasicBlock, SEBottleNeck]] = BasicBlock,
                 block_list: tuple = (3, 3, 3),
                 class_num: int = 1000,
                 channels: tuple = (16, 32, 64),
                 # norm_layer: nn.Module = nn.BatchNorm2d,
                 groups: int = 1,
                 base_width: int = 64):
        """
        ResNet32 Module Definition

        :param block: resnet architecture block
        :param block_list: number of block list
        :param class_num: number of class
        :param channels: number of channel for each block
        :param norm_layer: normalize layer
        :param groups: group for bottleNeck architecture
        :param base_width: base width for bottleNeck architecture
        """
        super(ResNet32, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.groups = groups
        self.base_width = base_width
        self.norm_layer = norm_layer
        self.in_channels = channels[0]

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=(7, 7), stride=1, padding=(1, 1), bias=False)
        self.bn1 = self.norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[-1] * block.factor, class_num)

        self.layers = [self.make_layer(block=block, nblock=block_list[i], channels=channels[i]) for i in range(len(block_list))]
        self.register_layer()

    def features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_impl(self, x):
        x = self.features(x)
        return self.fc(self.flatten(self.avgpool(x)))

    def predict(self, x):
        x = self.features(x)
        return self.fc(self.flatten(self.avgpool(x)))

    def forward(self, *args):
        return self.forward_impl(*args) if self.training else self.predict(*args)

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


def get_resnet32(model_name: str, nclass=1000, zero_init_residual=False, dataset=None, **kwargs) -> nn.Module:
    if model_name == 'resnet32_20':
        model = ResNet32(block=BasicBlock, block_list=[3, 3, 3], class_num=nclass)
    elif model_name == 'resnet32_110':
        model = ResNet32(block=BasicBlock, block_list=[18, 18, 18], class_num=nclass)

    resnet_normal_init(model)
    resnet_zero_init(model, zero_init_residual)

    if dataset:
        pass

    return model
