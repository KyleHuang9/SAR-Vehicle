import torch.nn as nn
from module.model.common import Classify


class Model(nn.Module):
    def __init__(self, nc, base_channels):
        super().__init__()
        self.model = Classify(nc=nc, base_channels=base_channels)

    def forward(self, x):
        output = self.model(x)
        return output