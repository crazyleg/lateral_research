import torch
import itertools

from torch import nn
from torch.nn import functional as fn


class LateralConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, feature_map_size):
        super(LateralConv2d, self).__init__(in_channels, out_channels, kernel_size)

        locations = feature_map_size[0] * feature_map_size[1]
        self.laterals = nn.Parameter(torch.Tensor(out_channels, locations, locations))
        self._feature_maps = []
        self.is_laterals_calculated = False
        self.is_lateran_training = False

    def calculate_laterals(self):
        feature_maps = torch.cat(self._feature_maps)
        print("feature maps: ", feature_maps.shape)

        out_channels, width, height = feature_maps.shape[1:]
        assert out_channels == self.laterals.shape[0]
        assert width == self.laterals.shape[1]
        assert height == self.laterals.shape[2]

        # normalize feature maps so that at every spatial location n, the sum over all filters is unity
        # ...

        self._feature_maps = []
        self.is_laterals_calculated = True

    def forward(self, x):
        y = super(LateralConv2d, self).forward(x)

        if self.is_lateran_training:
            self._feature_maps.append(y.detach())

        if self.is_laterals_calculated:
            ...  # add laterals

        return y


class LateralCNN(nn.Module):
    def __init__(self, in_channels):
        super(LateralCNN, self).__init__()

        self.in_channels = in_channels

        # W(10, 1, 5, 5)
        self.conv1 = LateralConv2d(self.in_channels, 10, kernel_size=(5, 5), feature_map_size=(24, 24))
        # W(20, 10, 5, 5)
        self.conv2 = LateralConv2d(10, 20, kernel_size=(5, 5), feature_map_size=(8, 8))

        self.lin_out = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        x = fn.relu(self.conv1(x))
        x = fn.max_pool2d(x, (2, 2), (2, 2))
        x = fn.relu(self.conv2(x))
        x = fn.max_pool2d(x, (2, 2), (2, 2))
        x = x.flatten(1, -1)
        x = self.lin_out(x)
        return x

    def calculate_laterals(self):
        self.conv1.calculate_laterals()
        self.conv2.calculate_laterals()

    def collect_feature_maps(self):
        self.conv1.is_lateran_training = True
        self.conv2.is_lateran_training = True
