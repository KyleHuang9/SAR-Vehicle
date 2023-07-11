import torch
import torch.nn as nn

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "hardswish":
        module = nn.Hardswish(inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, act="relu"):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.fc(x)))

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class Classify(nn.Module):
    """number classify model"""
    
    def __init__(self, nc, base_channels):
        super().__init__()
        self.stem = BaseConv(in_channels=1, out_channels=base_channels // 2, ksize=3, stride=1) # /1
        self.block1 = nn.Sequential(
                    BaseConv(in_channels=base_channels // 2, out_channels=base_channels, ksize=3, stride=1),
                    BaseConv(in_channels=base_channels, out_channels=base_channels * 2, ksize=3, stride=2),
        ) # /2
        self.block2 = nn.Sequential(
                    BaseConv(in_channels=base_channels * 2, out_channels=base_channels * 2, ksize=3, stride=1),
                    BaseConv(in_channels=base_channels * 2, out_channels=base_channels * 4, ksize=3, stride=2),
        ) # /4
        self.block3 = nn.Sequential(
                    BaseConv(in_channels=base_channels * 4, out_channels=base_channels * 4, ksize=3, stride=1),
                    BaseConv(in_channels=base_channels * 4, out_channels=base_channels * 8, ksize=3, stride=2),
        ) # /8
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = BaseLinear(in_features=base_channels * 8, out_features=base_channels * 2)
        self.fc2 = nn.Linear(base_channels * 2, nc)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        output = self.fc2(x)
        return output