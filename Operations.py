import math

import torch
import torch.nn as nn


class ZeroOp(nn.Module):
    def __init__(self, stride=1):
        super(ZeroOp, self).__init__()
        self.stride = stride

    def forward(self, batched_images):
        dim = math.floor(batched_images.shape[-1] / self.stride)
        return batched_images[:, :, :dim, :dim].mul(0.)


OPERATIONS = {}
OPERATIONS[0] = ZeroOp
OPERATIONS[1] = nn.MaxPool2d
OPERATIONS[2] = nn.AvgPool2d
OPERATIONS[3] = nn.Conv2d
OPERATIONS[4] = nn.Identity


class FactorizedReduce(nn.Module):
    def __init__(self, in_channels, out_channels, affine=True):
        super(FactorizedReduce, self).__init__()
        self.activation = nn.LeakyReLU()
        self.conv_1 = nn.Conv2d(in_channels, math.floor(out_channels / 2), 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(in_channels, math.floor(out_channels / 2), 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, affine=affine)

    def forward(self, x):
        x = self.activation(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class ChannelFixer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, affine=True):
        super(ChannelFixer, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels, affine=affine)
        )

    def forward(self, x):
        return self.op(x)
