import torch
import pytorch_lightning as pl

from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule

from .model import HebbMLP
from .train_pipe import TrainPipe

device_id = None

latest_checkpoint = ""

if __name__ == '__main__':
    data_dir = '../data/'
    train_transforms = None
    val_transforms = None
    mnist_dm = MNISTDataModule(data_dir, train_transforms=train_transforms, val_transforms=val_transforms)
    mlp = HebbMLP(1)
    train_pipe = TrainPipe(model=mlp)

    classic_trainer = pl.Trainer(gpus=([device_id] if device_id is not None else None), max_epochs=10)
    print("TRAINING MLP")
    classic_trainer.fit(train_pipe, datamodule=mnist_dm)

    print("SAVE MLP")
    torch.save(train_pipe.state_dict(), 'lateral_model_v1')

    # print("TEST MLP")
    # classic_trainer.test(train_pipe, datamodule=mnist_dm)
