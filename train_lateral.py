import pandas as pd
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

    PATH = './cifar_net_n06.pth'
    net.load_state_dict(torch.load(PATH,  map_location=torch.device(device)))
    net.eval()
    net.set_lateral_mode(True)
    with torch.no_grad():

        for i, data in tqdm(enumerate(trainloader, 0)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            correct = 0
            total = 0
        net.eval()
        net.set_learned(True)
        net.set_lateral_mode(False)

        net.process_lateral()
    PATH = './cifar_net_l1_n06_normalized.pth'
    torch.save(net.state_dict(), PATH)
   # with torch.no_grad():
     #   for data in testloader:
      #      images, labels = data
       #     images, labels = images.to(device), labels.to(device)
        #    outputs = net(images)
         #   _, predicted = torch.max(outputs.data, 1)
#            total += labels.size(0)
 #           correct += (predicted == labels).sum().item()

  #      print('Accuracy of the network on the 10000 test images: %d %%' % (
   #             100 * correct / total))

