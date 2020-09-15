import torch
import pytorch_lightning as pl

from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule

from .model import DeepELM
from .train_pipe import TrainPipe

device_id = None

latest_checkpoint = ""

if __name__ == '__main__':
    train_batch_size = 1024
    val_batch_size = 2048
    data_dir = '../data/'
    train_transforms = None
    val_transforms = None
    mnist_dm = MNISTDataModule(data_dir, train_transforms=train_transforms, val_transforms=val_transforms)
    mlp = DeepELM(1)
    train_pipe = TrainPipe(model=mlp)

    classic_trainer = pl.Trainer(gpus=([device_id] if device_id is not None else None), max_epochs=1000)
    print("TRAINING MLP")
    classic_trainer.fit(
        train_pipe,
        train_dataloader=mnist_dm.train_dataloader(train_batch_size),
        val_dataloaders=mnist_dm.val_dataloader(val_batch_size))

    # print("SAVE MLP")
    # torch.save(train_pipe.state_dict(), 'model_v1')

    # print("TEST MLP")
    # classic_trainer.test(train_pipe, datamodule=mnist_dm)
