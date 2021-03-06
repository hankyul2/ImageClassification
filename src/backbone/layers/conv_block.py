import copy
import math
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False, conv=nn.Conv2d):
    return conv(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=bias, groups=groups)


def conv3x3(in_channels, out_channels, stride=1, groups=1, padding=1, conv=nn.Conv2d):
    return conv(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=padding, bias=False, groups=groups)


def conv5x5(in_channels, out_channels, stride=1, groups=1, padding=1, conv=nn.Conv2d):
    return conv(in_channels, out_channels, kernel_size=(5, 5), stride=stride, padding=padding, bias=False, groups=groups)


class BasicBlock(nn.Module):
    factor = 1

    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64):
        super(BasicBlock, self).__init__()
        self.out_channels = out_channels
        self.conv1 = conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3(in_channels=out_channels, out_channels=out_channels)
        self.bn1 = norm_layer(out_channels)
        self.bn2 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.downsample(x) + self.bn2(self.conv2(out)))


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

    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64, conv=nn.Conv2d):
        super(BottleNeck, self).__init__()
        self.width = width = int(out_channels * (base_width / 64.0)) * groups
        self.out_channels = out_channels * self.factor
        self.conv1 = conv1x1(in_channels, width, conv=conv)
        self.conv2 = conv3x3(width, width, stride, groups=groups, conv=conv)
        self.conv3 = conv1x1(width, self.out_channels, conv=conv)
        self.bn1 = norm_layer(width)
        self.bn2 = norm_layer(width)
        self.bn3 = norm_layer(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return self.relu(self.downsample(x) + self.bn3(self.conv3(out)))


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


class StochasticDepth(nn.Module):
    def __init__(self, prob, mode):
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x):
        if self.prob == 0.0 or not self.training:
            return x
        else:
            shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == 'row' else [1]
            return x * torch.empty(shape).bernoulli_(self.survival).div_(self.survival).to(x.device)


class ConvBNAct(nn.Sequential):
    """This is made following torchvision works"""
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1,
                 conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, act=nn.ReLU):
        super(ConvBNAct, self).__init__(
            conv_layer(in_channel, out_channel, kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=groups, bias=False),
            norm_layer(out_channel),
            act()
        )


class MBConvConfig:
    """Mobile Conv stage configuration used for MobileNet_v(2,3), EfficientNet"""
    def __init__(self, expand_ratio, kernel, stride, in_ch, out_ch, layers,
                 depth_mult=1.0, width_mult=1.0,
                 act=nn.ReLU6, norm_layer=nn.BatchNorm2d,
                 use_se=True, se_act1=nn.ReLU, se_act2=nn.Sigmoid, se_reduction_ratio=4, se_reduce_mode='adjust',
                 fused=False):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_ch = self.adjust_channels(in_ch, width_mult)
        self.out_ch = self.adjust_channels(out_ch, width_mult)
        self.num_layers = self.adjust_depth(layers, depth_mult)

        self.act = act
        self.norm_layer = norm_layer

        self.use_se = use_se
        self.se_act1 = se_act1
        self.se_act2 = se_act2
        self.se_reduction_ratio = se_reduction_ratio
        self.se_reduce_mode = se_reduce_mode

        self.fused = fused

    @property
    def se_reduce(self):
        if self.se_reduce_mode == 'adjust':
            return self.adjust_channels(self.in_ch, self.expand_ratio / self.se_reduction_ratio)
        else:
            return self.in_ch / self.se_reduction_ratio

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
    def adjust_channels(channel, factor, divisible=8):
        new_channel = channel * factor
        divisible_channel = max(divisible, (int(new_channel + divisible / 2) // divisible) * divisible)
        divisible_channel += divisible if divisible_channel < 0.9 * new_channel else 0
        return divisible_channel


class MBConv(nn.Module):
    """MobileNet_v2 main building blocks (from torchvision)"""
    def __init__(self, config, sd_prob=0.0):
        super(MBConv, self).__init__()
        inter_channel = config.adjust_channels(config.in_ch, config.expand_ratio)
        layers = []
        if config.expand_ratio != 1:
            layers.append(ConvBNAct(config.in_ch, inter_channel, kernel_size=1, stride=1, norm_layer=config.norm_layer, act=config.act))
        layers.append(ConvBNAct(inter_channel, inter_channel, kernel_size=config.kernel, stride=config.stride, groups=inter_channel, norm_layer=config.norm_layer, act=config.act))
        layers.append(conv1x1(inter_channel, config.out_ch, stride=1))
        layers.append(config.norm_layer(config.out_ch))
        self.conv = nn.Sequential(*layers)

        self.inter_channel = inter_channel
        self.use_skip_connection = config.stride == 1 and config.in_ch == config.out_ch
        self.stochastic_path = StochasticDepth(sd_prob, "row")

    def forward(self, x):
        out = self.conv(x)
        if self.use_skip_connection:
            out = x + self.stochastic_path(out)
        return out


class MBConvSE(MBConv):
    """EfficientNet main building blocks (from torchvision & timm works)"""
    def __init__(self, config, sd_prob=0.0):
        super(MBConvSE, self).__init__(config, sd_prob)

        if config.fused:
            block = [ConvBNAct(config.in_ch, self.inter_channel, kernel_size=config.kernel, stride=config.stride,norm_layer=config.norm_layer, act=config.act)]
        else:
            block = [*copy.deepcopy(self.conv[:-2])]

        if config.use_se:
            block.append(SEUnit(self.inter_channel, config.se_reduce, act1=config.se_act1, act2=config.se_act2))
        block.append(ConvBNAct(self.inter_channel, config.out_ch, kernel_size=1, stride=1, norm_layer=config.norm_layer, act=nn.Identity))
        self.block = nn.Sequential(*block)
        del self.conv

    def forward(self, x):
        out = self.block(x)
        if self.use_skip_connection:
            out = x + self.stochastic_path(out)
        return out


class SEUnit(nn.Module):
    def __init__(self, in_channel, hidden_dim=None, reduction_ratio=16, act1=nn.ReLU, act2=nn.Sigmoid):
        super(SEUnit, self).__init__()
        hidden_dim = int(hidden_dim if hidden_dim else in_channel // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = conv1x1(in_channel, hidden_dim, bias=True)
        self.fc2 = conv1x1(hidden_dim, in_channel, bias=True)
        self.act1 = act1()
        self.act2 = act2()

    def forward(self, x):
        return x * self.act2(self.fc2(self.act1(self.fc1(self.avg_pool(x)))))


class SEBasicBlock(BasicBlock):
    def __init__(self, *args, reduction_ratio=16, **kwargs):
        super(SEBasicBlock, self).__init__(*args, **kwargs)
        self.se_module = SEUnit(self.out_channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.downsample(x) + self.se_module(self.bn2(self.conv2(out))))


class SEBottleNeck(BottleNeck):
    def __init__(self, *args, reduction_ratio=16, **kwargs):
        super(SEBottleNeck, self).__init__(*args, **kwargs)
        self.se_module = SEUnit(self.out_channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return F.relu(self.downsample(x) + self.se_module(self.bn3(self.conv3(out))))


class StdConv(nn.Conv2d):
    def forward(self, x):
        return self._conv_forward(x, self.standarize(self.weight), self.bias)

    def standarize(self, x):
        return (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + 1e-6)


def resnet_normal_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)


def mobilenet_v2_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.zeros_(m.bias)


def resnet_zero_init(model, zero_init_residual):
    for m in model.modules():
        if isinstance(m, BottleNeck) and zero_init_residual:
            nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock) and zero_init_residual:
            nn.init.constant_(m.bn2.weight, 0)