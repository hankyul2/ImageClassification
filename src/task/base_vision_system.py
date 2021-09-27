import pytorch_lightning as pl
from torch import nn
from torch.optim import SGD
from torchmetrics import MetricCollection, Accuracy

from src.lr_schedulers import CosineLR
from src.utils import accuracy


class BaseVisionSystem(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 batch_step: int,
                 max_epoch: int = 100,
                 lr: float = 3e-2,
                 momentum: float = 0.95,
                 weight_decay: float = 0.0005):
        """
        Base Vision System

        :param model: backbone model
        :param batch_step: number of batch step in one epoch
        :param max_epoch: number of epoch
        :param lr: learning rate
        :param momentum: optimizer momentum
        :param weight_decay: optimizer weight decay
        """
        super(BaseVisionSystem, self).__init__()
        self.save_hyperparameters()
        self.model = self.hparams['model']
        self.criterion = nn.CrossEntropyLoss()

        metrics = MetricCollection({'top@1': Accuracy(top_k=1), 'top@5': Accuracy(top_k=5)})
        self.train_metric = metrics.clone(prefix='train/')
        self.valid_metric = metrics.clone(prefix='valid/')
        self.test_metric = metrics.clone(prefix='test/')

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        metric = self.train_metric(y_hat, y)
        self.log_dict({'train/loss': loss})
        self.log_dict(metric)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        metric = self.valid_metric(y_hat, y)
        self.log_dict({'valid/loss': loss}, add_dataloader_idx=True)
        self.log_dict(metric, add_dataloader_idx=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        metric = self.test_metric(y_hat, y)
        self.log_dict({'test/loss': loss}, add_dataloader_idx=True)
        self.log_dict(metric, add_dataloader_idx=True)
        return loss

    def configure_optimizers(self):
        optimizer = SGD([
            {'params': list(set(param for name, param in self.model.named_parameters() if 'fc' in name)), 'lr': self.hparams['lr']},
            {'params': list(set(param for name, param in self.model.named_parameters() if 'fc' not in name)), 'lr': self.hparams['lr'] * 0.1},
        ], momentum=self.hparams['momentum'], weight_decay=self.hparams['weight_decay'])

        lr_scheduler = {'scheduler': CosineLR(optimizer, niter=self.hparams['max_epoch'] * self.hparams['batch_step'], warmup=self.hparams['batch_step']), 'interval': 'step'}
        return {'optimizer': optimizer, 'scheduler': lr_scheduler}


