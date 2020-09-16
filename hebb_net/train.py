import torch
import os
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import CSVLogger
import neptune

import pytorch_lightning as pl

from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule

from model import HebbMLP
from train_pipe import TrainPipe

device_id = None if torch.cuda.device_count() == 0 else 0

latest_checkpoint = ""

@hydra.main(config_path="config", config_name="config")
def run(cfg: DictConfig):
    data_dir = '../data/'
    train_transforms = None
    val_transforms = None

    loggers = []
    csv_logger = CSVLogger("logs", name=cfg.experiment.name)
    loggers.append(csv_logger)
    if bool(cfg.batch_norm.use_batch_norm):
        cfg.experiment.name += cfg.batch_norm.name

    tags = [cfg.network.linear_module.name, 'Batch_use: '+str(cfg.batch_norm.use_batch_norm), cfg.experiment.optimizer,
            cfg.experiment.lr]
    if cfg.batch_norm.use_batch_norm:
        tags.append(cfg.batch_norm.name)

    neptune.init('crazyleg11/feedback-normalizations')
    neptune.create_experiment(name=cfg.experiment.name, tags=tags)

    mnist_dm = CIFAR10DataModule(data_dir, train_transforms=train_transforms, val_transforms=val_transforms)
    mlp = HebbMLP(1, cfg=cfg)
    train_pipe = TrainPipe(model=mlp, cfg=cfg)

    classic_trainer = pl.Trainer(gpus=([device_id] if device_id is not None else None),
                                 max_epochs=cfg.experiment.max_epochs,
                                 fast_dev_run=bool(cfg.experiment.fast_dev_run),
                                 logger=loggers)
    print("TRAINING MLP")
    classic_trainer.fit(train_pipe, datamodule=mnist_dm)

    print("SAVE MLP")
    torch.save(train_pipe.state_dict(), 'lateral_model_v1')

    # print("TEST MLP")
    # classic_trainer.test(train_pipe, datamodule=mnist_dm)


if __name__ == '__main__':
    run()
