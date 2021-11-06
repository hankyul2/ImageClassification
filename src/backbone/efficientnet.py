import math

from torch import nn


class MBConvConfig:
    def __init__(self, expand_ratio, kernel, stride, in_ch, out_ch, layers, depth_mult, width_mult):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_ch = self.adjust_channels(in_ch, width_mult)
        self.out_ch = self.adjust_channels(out_ch, width_mult)
        self.num_layers = self.adjust_depth(layers, depth_mult)

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
    def adjust_channels(channel, factor):
        new_channel = channel * factor
        divisible_channel = max(8, (int(new_channel + 4) // 8) * 8)
        divisible_channel += 8 if divisible_channel < 0.9 * new_channel else 0
        return divisible_channel
