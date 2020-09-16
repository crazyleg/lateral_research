import torch
import pytorch_lightning as pl
import neptune
from torch import nn, optim

class TrainPipe(pl.LightningModule):
    def __init__(self, *args, model):
        super(TrainPipe, self).__init__()

        self.model = model

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=1e-2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        accuracy = (torch.argmax(y_pred, 1) == y).float().mean()

        result = pl.TrainResult(loss)
        neptune.log_metric('loss/train', loss)
        neptune.log_metric('accuracy/train', accuracy)
        result.log('loss/train', loss, prog_bar=True, on_epoch=True)
        result.log('accuracy/train', accuracy, prog_bar=True, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        accuracy = (torch.argmax(y_pred, 1) == y).float().mean()

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('loss/val', loss, prog_bar=True)
        result.log('accuracy/val', accuracy, prog_bar=True)
        return result

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
