import copy
import torch
import numpy as np
import pytorch_lightning as pl

from torch import optim
from torch.nn import functional as fn
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule

from .model import DualClassifier, ReversedSiameseNet

device_id = 1
latest_checkpoint = ""


def make_hook(saved_grad):
    def _wrap(grad):
        saved_grad.append(grad)
        return grad

    return _wrap


if __name__ == '__main__':
    epochs = 30
    train_batch_size = 1024
    val_batch_size = 2048
    net_opt_lr = 1e-1
    siamese_net_opt_lr = 1e-3
    data_dir = '../data/'
    train_transforms = None
    val_transforms = None
    mnist_dm = MNISTDataModule(data_dir, train_transforms=train_transforms, val_transforms=val_transforms)

    net = DualClassifier().cuda(device_id)
    net_opt = optim.SGD(net.parameters(), lr=net_opt_lr)

    siamese_net = ReversedSiameseNet().cuda(device_id)
    siamese_net_opt = optim.Adam(siamese_net.parameters(), lr=siamese_net_opt_lr)

    for eid in range(epochs):

        train_accuracy, train_loss = [], []
        for batch_id, batch in enumerate(mnist_dm.train_dataloader(train_batch_size)):
            x, y_true = batch[0].cuda(device_id), batch[1].cuda(device_id)

            # train_net = batch_id % 5 == 0

            # collect per-layer inputs and y_pred gradient
            net_opt.zero_grad()
            y_pred, inputs = net(x, siamese_net)
            saved_grad = []
            y_pred.register_hook(make_hook(saved_grad))
            loss = fn.cross_entropy(y_pred, y_true)
            loss.backward()

            # print(net.l1.forward_weight.grad[0, 0:10])
            # print(net.l2.forward_weight.grad[0, 0:10])
            # print(net.l3.forward_weight.grad[0, 0:10])
            net_opt.zero_grad()

            # calculate gradients for net
            siamese_net.train()
            siamese_net_opt.zero_grad()
            grads = siamese_net(saved_grad[0], inputs)
            # for g in grads:
            #     print(g[0, 0:10])
            # exit(0)

            # copy net and apply gradient
            _net = copy.deepcopy(net)
            _net.eval()
            _net.requires_grad_(False)
            for p, d_p in zip(_net.parameters(), grads):
                p.add_(d_p, alpha=-net_opt_lr)

            # calculate net loss and update siamese_net weights
            y_pred, _ = _net(x, siamese_net)
            loss = fn.cross_entropy(y_pred, y_true)
            loss.backward()
            siamese_net_opt.step()

            net.train()
            siamese_net.eval()
            net_opt.zero_grad()
            y_pred, _ = net(x, siamese_net)
            loss = fn.cross_entropy(y_pred, y_true)
            train_loss.append(loss.detach().cpu().numpy())
            accuracy = (torch.argmax(y_pred, 1) == y_true).float().mean()
            train_accuracy.append(accuracy.detach().cpu().numpy())

            loss.backward()
            net_opt.step()

        print(f"{eid} | Train: loss={np.mean(train_loss):.6f}, acc={np.mean(train_accuracy):.4f}")

        net.eval()
        val_accuracy, val_loss = [], []
        for batch in mnist_dm.val_dataloader(val_batch_size):
            x, y_true = batch[0].cuda(device_id), batch[1].cuda(device_id)

            y_pred, _ = net(x, siamese_net)
            loss = fn.cross_entropy(y_pred, y_true)
            val_loss.append(loss.detach().cpu().numpy())
            accuracy = (torch.argmax(y_pred, 1) == y_true).float().mean()
            val_accuracy.append(accuracy.detach().cpu().numpy())

        print(f"{eid} | Val: loss={np.mean(val_loss):.6f}, acc={np.mean(val_accuracy):.4f}")
