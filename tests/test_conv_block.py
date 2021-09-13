import torch
from torch import nn

from src.model.layers.conv_block import InvertedResidualBlock


def test_inverted_residual_block_forward():
    x = torch.rand((4, 16, 32, 32))
    IRBlock = InvertedResidualBlock(factor=6, in_channels=16, out_channels=16, stride=1, norm_layer=nn.BatchNorm2d,
                                    downsample=None, dropout=0.1)
    out = IRBlock(x)
    assert list(out.shape) == [4, 16, 32, 32]
