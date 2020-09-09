import pytorch_lightning as pl

from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule

from .model import LateralCNN
from .train_pipe import TrainPipe

device_id = 1

# latest_checkpoint = "lightning_logs/version_0/checkpoints/epoch=9.ckpt"
latest_checkpoint = ""

if __name__ == '__main__':
    data_dir = '../data/'
    train_transforms = None
    val_transforms = None
    mnist_dm = MNISTDataModule(data_dir, train_transforms=train_transforms, val_transforms=val_transforms)
    lt_cnn_model = LateralCNN(1)

    classic_trainer = pl.Trainer(gpus=([device_id] if device_id else None), max_epochs=1)
    if not latest_checkpoint:
        print("TRAINING CLASSIC CNN")
        train_pipe = TrainPipe(model=lt_cnn_model)
        classic_trainer.fit(train_pipe, datamodule=mnist_dm)

        print("TEST CLASSIC CNN")
        classic_trainer.test(train_pipe, datamodule=mnist_dm)
    else:
        print(f"LOADING FROM CHECKPOINT: {latest_checkpoint}")
        train_pipe = TrainPipe.load_from_checkpoint(latest_checkpoint, model=lt_cnn_model)

    print('COLLECTING FEATURE MAPS')
    train_pipe.start_lateral_training()
    for batch in mnist_dm.train_dataloader():
        if device_id is not None:
            batch = (batch[0].cuda(1), batch[1].cuda(1))

        train_pipe.validation_step(batch, None)

    print("CALCULATING LATERAL LAYERS")
    train_pipe.finish_lateral_training()

    print("TEST CNN WITH LATERAL CONNECTIONS")
    classic_trainer.test(train_pipe, datamodule=mnist_dm)
