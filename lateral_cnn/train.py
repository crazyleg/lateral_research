import torch
import pytorch_lightning as pl

from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule

from .model import LateralCNN
from .train_pipe import TrainPipe

device_id = None

# latest_checkpoint = "lateral_model_v1"
latest_checkpoint = ""

if __name__ == '__main__':
    data_dir = '../data/'
    train_transforms = None
    val_transforms = None
    mnist_dm = MNISTDataModule(data_dir, train_transforms=train_transforms, val_transforms=val_transforms)
    lt_cnn_model = LateralCNN(1)
    train_pipe = TrainPipe(model=lt_cnn_model)

    classic_trainer = pl.Trainer(gpus=([device_id] if device_id is not None else None), max_epochs=10)
    if not latest_checkpoint:
        print("TRAINING CLASSIC CNN")
        classic_trainer.fit(train_pipe, datamodule=mnist_dm)

        print('COLLECTING FEATURE MAPS')
        train_pipe.start_lateral_training()
        for batch in mnist_dm.train_dataloader():
            if device_id is not None:
                batch = (batch[0].cuda(device_id), batch[1].cuda(device_id))

            train_pipe.validation_step(batch, None)

        print("CALCULATING LATERAL LAYERS")
        train_pipe.finish_lateral_training()

        print("SAVE LATERAL CNN")
        torch.save(train_pipe.state_dict(), 'lateral_model_v1')
    else:
        train_pipe.load_state_dict(torch.load(latest_checkpoint))
        train_pipe.enable_laterals()

    print(train_pipe.model.conv1.laterals)
    print("TEST CLASSIC CNN")
    train_pipe.disable_laterals()
    classic_trainer.test(train_pipe, datamodule=mnist_dm)

    print("TEST CNN WITH LATERAL CONNECTIONS")
    train_pipe.enable_laterals()
    classic_trainer.test(train_pipe, datamodule=mnist_dm)
