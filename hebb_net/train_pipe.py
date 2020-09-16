import torch
import sys
import pytorch_lightning as pl
import neptune
from torch import nn
from torch.optim import Adam, SGD

def str_to_class(str):
    return getattr(sys.modules[__name__], str)

class TrainPipe(pl.LightningModule):
    def __init__(self, *args, model, cfg):
        super(TrainPipe, self).__init__()

        self.model = model
        self.cfg = cfg

    def configure_optimizers(self):
        return str_to_class(self.cfg.experiment.optimizer)(self.parameters(), lr=self.cfg.experiment.lr)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        accuracy = (torch.argmax(y_pred, 1) == y).float().mean()

        result = pl.TrainResult(loss)
        result.log('loss/train', loss, prog_bar=True, on_epoch=True)
        result.log('accuracy/train', accuracy, prog_bar=True, on_epoch=True)
        for logger in self.logger:
            logger.experiment.log_metrics({'loss/train':loss,'accuracy/train':accuracy})
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        accuracy = (torch.argmax(y_pred, 1) == y).float().mean()

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('loss/val', loss, prog_bar=True)
        result.log('accuracy/val', accuracy, prog_bar=True)
        for logger in self.logger:
            logger.experiment.log_metrics({'loss/val': loss, 'accuracy/val': accuracy})
        return result

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
