import torch
import hydra
from omegaconf import DictConfig
import neptune

# The init() function called this way assumes that
# NEPTUNE_API_TOKEN environment variable is defined.

import pytorch_lightning as pl

from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule

from hebb_net.model import HebbMLP
from hebb_net.train_pipe import TrainPipe

device_id = None

latest_checkpoint = ""

@hydra.main(config_name="config/config.yaml")
def run(cfg: DictConfig):
    data_dir = '../data/'
    train_transforms = None
    val_transforms = None


    if cfg.experiment.name is not None:
        neptune.set_project('crazyleg11/feedback-normalizations')
        neptune.create_experiment(name='C10_baseline')
    mnist_dm = CIFAR10DataModule(data_dir, train_transforms=train_transforms, val_transforms=val_transforms)
    mlp = HebbMLP(1)
    train_pipe = TrainPipe(model=mlp)

    classic_trainer = pl.Trainer(gpus=([device_id] if device_id is not None else None), max_epochs=10)
    print("TRAINING MLP")
    classic_trainer.fit(train_pipe, datamodule=mnist_dm)

    print("SAVE MLP")
    torch.save(train_pipe.state_dict(), 'lateral_model_v1')

    # print("TEST MLP")
    # classic_trainer.test(train_pipe, datamodule=mnist_dm)


if __name__ == '__main__':
    run()
