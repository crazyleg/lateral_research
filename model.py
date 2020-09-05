from typing import Any

import torch
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import nn
import numpy as np
import torch.nn.functional as F

class Conv2DWithLateralFeatures(nn.Module):
    def __init__(self, *args, **kwags):
        super(Conv2DWithLateralFeatures, self).__init__()
        self.lateral_mode = args[3]
        self.output_layers = args[1]
        self.alpha = 0.1
        self.lateral = torch.nn.parameter.Parameter(torch.ones(args[1], args[1]), requires_grad=False)
        self.lateral_storage = []
        self.conv = nn.Conv2d(*args[0:3])
        self.learned = False

    def set_lateral_mode(self, mode):
        self.lateral_mode = mode

    def set_leared_mode(self, mode=True):
        self.learned = mode

    def process_lateral(self):
        self.lateral = torch.nn.parameter.Parameter(torch.from_numpy(np.array(self.lateral_storage).mean(axis=0)), requires_grad=False)

    def forward(self, x):
        x = self.conv(x)
        # print('f ', x.shape)
        if self.lateral_mode:
            corrs = np.corrcoef(x.cpu().numpy().transpose((1,0,2,3)).reshape(self.output_layers,-1))
            self.lateral_storage.append(corrs)
            return x

        if self.learned:
            for i in range(self.output_layers):
                tmp = self.lateral[i]
                tmp[i] = 0
                x += self.alpha*(tmp.view((1,self.output_layers,1,1)) * x)
                # x[:, i, ...] = (x.mean(axis=(0, 2, 3)) * tmp).sum()
            return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lateral_mode = False
        self.learned = False

        self.conv1 = Conv2DWithLateralFeatures(3, 10, 5, self.lateral_mode)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = Conv2DWithLateralFeatures(10, 20, 5, self.lateral_mode)
        self.fc1 = nn.Linear(20 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 10)

    def set_lateral_mode(self, mode=False):
        self.lateral_mode = mode
        self.conv1.set_lateral_mode(mode)
        self.conv2.set_lateral_mode(mode)

    def set_learned(self, mode=True):
        self.learned = mode
        self.conv1.learned = mode
        self.conv2.learned = mode

    def process_lateral(self):
        self.conv1.process_lateral()
        self.conv2.process_lateral()
        pass

    def forward(self, x):
        # print(self.conv1(x).shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
