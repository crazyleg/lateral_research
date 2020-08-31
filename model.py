from typing import Any

import torch
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

class Conv2DWithLateralFeatures(nn.Module):
    def __init__(self, *args, **kwags):
        super(Conv2DWithLateralFeatures, self).__init__()
        self.lateral_mode = args[3]
        self.output_layers = args[1]
        self.alpha = 0.001
        self.lateral = torch.nn.parameter.Parameter(torch.ones(args[1], args[1]), requires_grad=False)
        self.conv = nn.Conv2d(*args[0:3])
        self.learned = False

    def set_lateral_mode(self, mode):
        self.lateral_mode = mode

    def set_leared_mode(self, mode=True):
        self.learned = mode

    def forward(self, x):
        x = self.conv(x)
        # print('f ', x.shape)
        if self.lateral_mode:
            for i in range(self.output_layers):
                # print(x[:,i].abs().mean((1,2)).shape)
                # print(x.abs().mean((2,3)).shape)
                # a1 = x[:,i].abs().mean((1,2)).view(-1,1)*x.abs().mean((2,3))
                means = x.abs().mean((0,2,3))
                self.lateral[i] += means[i]*means/x.abs().std((0,2,3))
                # a1 = x[]
                # a1 = x[:, i].std() * x.std((0, 2, 3))
                # print((x[:,i].abs().mean((1,2)).view(-1,1)*x.abs().mean((2,3))).shape)
                # print((x[:,i].abs().mean((1,2))*x[:,i].abs().mean((1,2))).shape)
                # print(torch.einsum('bs,bs->b', x.abs().mean((2,3)), x.abs().mean((2,3))).shape)
                # a2 = x[:, i].std() ** 2
                # a3 = x.std((0, 2, 3)).std() ** 2
                # a2 = x[:,i].abs().mean((1,2)).view(-1,1)*x.abs().mean((2,3))
                # a3 = torch.einsum('bs,bs->b', x.abs().mean((2,3)), x.abs().mean((2,3)))
                # TODO delete self-excitation
                # print(a1.shape, a2.mввean(1).shape, a3.shape, (a2.mean(1)*a3).shape)
                # l_activations = a1 / (a2 * a3) / 12000
                # l_activations =  (a1 / ((a2.mean(1)*a3).view(-1,1))).mean(0)
                # self.lateral[i] = l_activations
                # print(l_activations.shape/)
                # todo zero out self-connections
                pass
            pass

        if self.learned:
            for i in range(self.output_layers):
                # print(self.lateral[i].shape, x.shape)
                l = self.lateral[i]
                l[i] = 0.

#         #         # tmp = self.lateral[i].unsqueeze(1).unsqueeze(1).unsqueeze(0).expand(
#         #         #    (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
#         #         # print(tmp.shape, x.shape)
#         #         # print((tmp*x*self.alpha).mean(1).shape, x[:,i].shape)
#         #         # TODO remove self-excitation
#         #         # TODO not mean but squared mean or somewhat std related
#         #         # TODO Dont do this during NN training
#         #         # print((self.alpha * l * x.abs().mean((0,2,3))).mean().view(1,1,1), x.abs().mean((0,2,3)))
#         #         x[:, i] = x[:, i] + (self.alpha * l.view(1,-1,1,1) * x).mean()
#         # return x
#         #
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.lateral_mode = False
#         self.learned = False
#
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

    def forward(self, x):
        # print(self.conv1(x).shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
