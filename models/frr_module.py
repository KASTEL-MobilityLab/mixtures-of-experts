"""
FRRN basic modules implementation
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as cp


FRRN_PARAMS = {
    "A": {
        "pool_stream_channel": 48,
        "res_stream_channel": 32,
        "encoder": [[3, 96, 2], [4, 192, 4], [2, 384, 8], [2, 384, 16]],
        "decoder": [[2, 192, 8], [2, 192, 4], [2, 96, 2]],
    },
    "B": {
        "pool_stream_channel": 48,
        "res_stream_channel": 32,
        "encoder": [[3, 96, 2], [4, 192, 4], [2, 384, 8], [2, 384, 16], [2, 384, 32]],
        "decoder": [[2, 192, 16], [2, 192, 8], [2, 192, 4], [2, 96, 2]],
    },
}


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def efficient_res_block(relu, norm1, conv1, norm2, conv2):
    """Efficient residual block based on the torch checkpoint"""
    def res_block_function(*inputs):
        out_temp = relu(norm1(conv1(*inputs)))
        out = relu(torch.add(norm2(conv2(out_temp)), *inputs))
        return out
    return res_block_function


def efficient_frr_block(relu, norm1, conv1, norm2, conv2):
    """Efficient FRR unit based on the torch checkpoint"""
    def frr_block_function(y_in, z_down):
        out_temp = relu(norm1(conv1(torch.cat((y_in, z_down), 1))))
        out = relu(norm2(conv2(out_temp)))
        return out
    return frr_block_function


class BasicBlock(nn.Module):
    """Basic residual block"""
    def __init__(self, inplanes, planes, stride=1, efficient=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.efficent = efficient

    def forward(self, x):
        res_block_function = efficient_res_block(self.relu, self.bn1, self.conv1,
                                                 self.bn2, self.conv2)
        out = cp(res_block_function, x) if self.efficent else res_block_function(x)
        return out


class FRRUnit(nn.Module):
    """Full-resolution residual unit (FRRUnit) """
    def __init__(self, y_in_c, y_out_c, z_c=32, factor=2, efficient=False):
        super(FRRUnit, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=factor, padding=1)
        self.conv1 = conv3x3(y_in_c+z_c, y_out_c)
        self.bn1 = nn.BatchNorm2d(y_out_c)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(y_out_c, y_out_c)
        self.bn2 = nn.BatchNorm2d(y_out_c)
        self.convz = nn.Conv2d(in_channels=y_out_c, out_channels=z_c, kernel_size=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=factor)
        self.efficient = efficient

    def forward(self, y_in, z_in):
        z_down = self.pool(z_in)
        frr_block_function = efficient_frr_block(self.relu, self.bn1, self.conv1,
                                                 self.bn2, self.conv2)
        y_out = cp(frr_block_function, y_in, z_down) \
            if self.efficient else frr_block_function(y_in, z_down)
        z_out = z_in + self.up(self.convz(y_out))
        return y_out, z_out


class FRRLayer(nn.Module):
    """Full-resolution residual layer (FRRLayer)"""
    def __init__(self, in_channels, setup, z_c=32, efficient=False):
        super(FRRLayer, self).__init__()
        num_blocks, out_channels, factor = setup
        self.frr1 = FRRUnit(in_channels, out_channels, z_c, factor, efficient)
        self.nexts = nn.ModuleList([FRRUnit(out_channels, out_channels, z_c, factor, efficient)
                                    for _ in range(1, num_blocks)])

    def forward(self, y_in, z_in):
        y_in, z_in = self.frr1(y_in, z_in)
        for frru_block in self.nexts:
            y_in, z_in = frru_block(y_in, z_in)
        return y_in, z_in
