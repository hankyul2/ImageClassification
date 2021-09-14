from torch import nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False, groups=groups)


class BasicBlock(nn.Module):
    factor = 1

    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3(in_channels=out_channels, out_channels=out_channels)
        self.bn1 = norm_layer(out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample if downsample else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.downsample(x) + self.bn2(self.conv2(out)))


class PreActBasicBlock(BasicBlock):
    factor = 1

    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64):
        super(BasicBlock, self).__init__(in_channels, out_channels, stride, norm_layer, downsample, groups, base_width)
        self.bn1 = norm_layer(in_channels)
        self.bn2 = norm_layer(in_channels)
        self.downsample = nn.Sequential(
                norm_layer(in_channels),
                conv1x1(in_channels, out_channels * self.factor, stride=stride),
            ) if downsample else nn.Identity()

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        return self.downsample(x) + F.relu(self.conv2(self.bn2(x)))


class BottleNeck(nn.Module):
    factor = 4

    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64):
        super(BottleNeck, self).__init__()
        self.width = width = int(out_channels * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(in_channels, width)
        self.conv2 = conv3x3(width, width, stride, groups=groups)
        self.conv3 = conv1x1(width, out_channels * self.factor)
        self.bn1 = norm_layer(width)
        self.bn2 = norm_layer(width)
        self.bn3 = norm_layer(out_channels * self.factor)
        self.downsample = downsample if downsample else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return F.relu(self.downsample(x) + self.bn3(self.conv3(out)))


class PreActBottleNeck(BottleNeck):
    '''Define ResNet Version2. If you want to apply PreActivation, you can just replace original BottleNeck to this'''

    factor = 4

    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64):
        super(PreActBottleNeck, self).__init__(in_channels, out_channels, stride, norm_layer, downsample, groups, base_width)
        self.bn1 = norm_layer(in_channels)
        self.bn3 = norm_layer(self.width)
        self.downsample = nn.Sequential(
                norm_layer(in_channels),
                conv1x1(in_channels, out_channels * self.factor, stride=stride),
            ) if downsample else nn.Identity()

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return self.downsample(x) + self.conv3(F.relu(self.bn3(out)))


class ConvBNReLU(nn.Sequential):
    """This is made following torchvision works"""
    def __init__(self, in_channel, out_channel, stride, conv_layer, norm_layer, act):
        super(ConvBNReLU, self).__init__(conv_layer(in_channel, out_channel, stride=stride), norm_layer(out_channel), act())


class InvertedResidualBlock(nn.Module):
    def __init__(self, factor, in_channels, out_channels, stride, norm_layer, act=nn.ReLU6):
        super(InvertedResidualBlock, self).__init__()
        inter_channel = in_channels * factor
        self.conv = nn.Sequential(
            ConvBNReLU(in_channels, inter_channel, 1, conv1x1, norm_layer, act),
            ConvBNReLU(inter_channel, inter_channel, stride, conv3x3, norm_layer, act),
            conv1x1(inter_channel, out_channels, stride=1), norm_layer(out_channels)
        )

        self.skip_connection = nn.Identity() if stride == 1 and in_channels == out_channels else lambda x: 0

    def forward(self, x):
        return self.skip_connection(x) + self.conv(x)


def resnet_normal_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)


def resnet_zero_init(model, zero_init_residual):
    for m in model.modules():
        if isinstance(m, BottleNeck) and zero_init_residual:
            nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock) and zero_init_residual:
            nn.init.constant_(m.bn2.weight, 0)