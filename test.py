import pandas as pd
from tqdm.auto import tqdm
import torch
from torch import optim
from torch import nn
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.autograd import Variable

from model import Net
USE_CUDA = True if torch.cuda.is_available() else False
device = torch.device("cuda" if USE_CUDA else "cpu")



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         AddGaussianNoise(0., 0.5)])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=16)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = Net()
    net = net.to(device)

    PATH = './cifar_net_l1_n06_normalized.pth'
    net.load_state_dict(torch.load(PATH,  map_location=torch.device(device)))
    net.eval()
    net.set_lateral_mode(False)
    net.set_learned(True)

    results = pd.DataFrame(columns=['alpha','mode','result'])
    for a in tqdm([0.8, 0.7, 0.6, 0.5, 0.4,
                   0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.003,0.002,
                   0.001, 0.0004, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.00002, 0.00008, 0.00005, 0.00001, 1e-6, 1e-7, 0]):
    #afr a in [0]:
        net.set_alpha(alpha=a)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            results = results.append({'alpha': a, 'mode': 'pixel-wise', 'result': 100* correct/total}, ignore_index=True)

            print('Accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct / total))
    results.to_csv('results_n05_trained_norm.csv')
