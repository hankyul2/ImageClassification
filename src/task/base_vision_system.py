import pytorch_lightning as pl
from torch import nn
from torch.optim import SGD

from src.lr_schedulers import CosineLR
from src.utils import accuracy


class BaseVisionSystem(pl.LightningModule):
    def __init__(self,
                 model: pl.LightningModule,
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
        # self.save_hyperparameters()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.batch_step = batch_step
        self.max_epoch = max_epoch
        self.momentum = momentum
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
        self.log('train_loss', loss, logger=True, sync_dist=True)
        self.log('train_acc1', acc1, logger=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc5', acc5, logger=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
        return loss, acc1, acc5

    def validation_step(self, batch, batch_idx):
        loss, acc1, acc5 = self.shared_eval_step(batch, batch_idx)
        self.log_dict({'val_loss':loss, 'val_acc1':acc1, 'val_acc5':acc5}, logger=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc1, acc5 = self.shared_eval_step(batch, batch_idx)
        self.log_dict({'test_loss': loss, 'test_acc1': acc1, 'test_acc5': acc5}, logger=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = SGD([
            {'params': list(set(param for name, param in self.model.named_parameters() if 'fc' in name)), 'lr': self.lr},
            {'params': list(set(param for name, param in self.model.named_parameters() if 'fc' not in name)), 'lr': self.lr * 0.1},
        ], momentum=self.momentum, weight_decay=self.weight_decay)

        lr_scheduler = {'scheduler': CosineLR(optimizer, niter=self.max_epoch * self.batch_step, warmup=self.batch_step), 'interval': 'step'}
        return {'optimizer': optimizer, 'scheduler': lr_scheduler}


