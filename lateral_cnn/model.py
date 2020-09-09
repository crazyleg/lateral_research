import torch
import itertools

from tqdm import tqdm
from torch import nn
from torch.nn import functional as fn


class LateralConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, feature_map_size):
        super(LateralConv2d, self).__init__(in_channels, out_channels, kernel_size)

        # To reduce the dimensions and size of this matrix, we assume translational invariance:
        #   only the relative position of two filters is important.
        # We limit it in space to three times the size of the classical receptive field, as the relative
        # co-occurrence probabilities decrease significantly above this scale.
        # locations = (kernel_size[0] * 3, kernel_size[1] * 3)

        # calc all combinations, for brevity
        self.locations = feature_map_size[0] * feature_map_size[1]

        # Positive weights correspond to excitatory connections, and negative weights to inhibition.
        # Excitatory neurons with similar orien-tation tuning connect to each other with
        # higher probability than to those tuned to the orthogonal orientation.
        #
        # laterals shape (source filter, target filter, source location, target location)
        self.laterals = nn.Parameter(torch.Tensor(out_channels, out_channels, self.locations, self.locations))

        self._feature_maps = []
        self.is_laterals_enabled = False
        self.is_lateran_training = False
        self.laterals_strength = 0.001

    def calculate_laterals(self):
        feature_maps = torch.cat(self._feature_maps)
        print("feature maps: ", feature_maps.shape)

        nsamples, out_channels, width, height = feature_maps.shape
        assert out_channels == self.laterals.shape[0]
        assert out_channels == self.laterals.shape[1]

        # rectify and normalize feature maps so that at every spatial location n, the sum over all filters is unity
        # feature_maps = torch.relu(feature_maps)
        feature_maps = torch.reshape(feature_maps, (nsamples, out_channels, width * height))

        # safe div to normalize feature maps
        # feature_maps = feature_maps / torch.sum(feature_maps, dim=1, keepdim=True)
        # feature_maps[feature_maps != feature_maps] = 0.

        # calculate lateral weights
        expected_iters = len(list(itertools.combinations_with_replacement(range(out_channels), 2)))
        for k, j in tqdm(itertools.combinations_with_replacement(range(out_channels), 2), total=expected_iters):
            for n in range(self.locations):
                fc_kn = feature_maps[:, k, n]
                fc_j = feature_maps[:, j, :]
                num = torch.sum(fc_kn[:, None] * fc_j, 0)
                den = torch.sum(fc_kn * fc_kn) * torch.sum(fc_j * fc_j, 0)
                val = num / den

                # zero diagonal
                val[n] = 0.

                self.laterals[j, k, n, :] = self.laterals[k, j, n, :] = val

        contain_nan = torch.isnan(self.laterals).any()
        if contain_nan:
            print("LATERALS CALCULATION MAY FAILED BECAUSE OF NAN VALUES")
            self.laterals[self.laterals != self.laterals] = 0.
        else:
            print("LATERALS CALCULATED SUCCESSFULLY")

        # XXX: wtf inhibition ??
        # self.laterals -= 1.

        self.is_laterals_enabled = True
        self._feature_maps = []

    def enable_laterals(self):
        self.is_laterals_enabled = True

    def disable_laterals(self):
        self.is_laterals_enabled = False

    def forward(self, x):
        y = super(LateralConv2d, self).forward(x)

        if self.is_lateran_training:
            self._feature_maps.append(y.detach())

        if self.is_laterals_enabled:
            y_shape = y.shape
            y = torch.reshape(y, (y_shape[0], self.out_channels, -1))
            for j in range(self.out_channels):
                for m in range(self.locations):
                    fc_jm = y[:, j, m]
                    y[:, j, m] = fc_jm + fc_jm * self.laterals_strength * torch.sum(self.laterals[j:j+1, :, m, :] * y)

            y = torch.reshape(y, y_shape)

        return y


class LateralCNN(nn.Module):
    def __init__(self, in_channels):
        super(LateralCNN, self).__init__()

        self.in_channels = in_channels
        self.conv1 = LateralConv2d(self.in_channels, 10, kernel_size=(5, 5), feature_map_size=(24, 24))
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

    def enable_laterals(self):
        self.conv1.enable_laterals()
        self.conv2.enable_laterals()

    def disable_laterals(self):
        self.conv1.disable_laterals()
        self.conv2.disable_laterals()
