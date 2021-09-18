import pytorch_lightning as pl
from torch import nn
from torch.optim import SGD

from src.lr_schedulers import CosineLR
from src.utils import accuracy


class ImageClassificationTask(pl.LightningModule):
    def __init__(self, model, niter, lr, momentum=0.95, weight_decay=0.0005):
        super(ImageClassificationTask, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.niter = niter
        self.momentum = momentum
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
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

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        pass

    def configure_optimizers(self):
        optimizer1 = SGD(list(set(param for name, param in self.model.named_parameters() if 'fc' in name)),
                         lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        optimizer2 = SGD(list(set(param for name, param in self.model.named_parameters() if 'fc' not in name)),
                         lr=self.lr*0.1, momentum=self.momentum, weight_decay=self.weight_decay)
        lr_scheduler1 = {'scheduler': CosineLR(optimizer1, niter=self.niter, lr=self.lr), 'interval': 'step'}
        lr_scheduler2 = {'scheduler': CosineLR(optimizer2, niter=self.niter, lr=self.lr*0.1), 'interval': 'step'}
        return [optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2]


