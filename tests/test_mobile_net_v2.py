import torch

from src.model.layers.conv_block import InvertedResidualBlock
from src.model.mobile_net_v2 import MobileNetV2


def test_mobile_net_v2():
    x = torch.rand((8, 3, 224, 224))
    model = MobileNetV2(block=InvertedResidualBlock, nclass=10)
    output = model(x)
    assert list(output.shape) == [8, 10]
