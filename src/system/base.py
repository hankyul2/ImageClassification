import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import instantiate_class
from torchmetrics import MetricCollection, Accuracy
from torch import nn

from src.backbone.models import get_model


class BaseVisionSystem(pl.LightningModule):
    def __init__(self, backbone_init: dict, num_classes: int, num_step: int, max_epochs: int,
                 optimizer_init: dict, lr_scheduler_init: dict):
        """ Define base vision classification system

        :arg
            backbone_init: feature extractor
            num_classes: number of class of dataset
            num_step: number of step
            max_epoch: max number of epoch
            optimizer_init: optimizer class path and init args
            lr_scheduler_init: learning rate scheduler class path and init args
        """
        super(BaseVisionSystem, self).__init__()

        # step 1. save data related info (not defined here)
        self.num_step = num_step
        self.max_epochs = max_epochs

        # step 2. define model
        self.backbone = get_model(**backbone_init)
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)

        # step 3. define lr tools (optimizer, lr scheduler)
        self.optimizer_init_config = optimizer_init
        self.lr_scheduler_init_config = lr_scheduler_init
        self.criterion = nn.CrossEntropyLoss()

        # step 4. define metric
        metrics = MetricCollection({'top@1': Accuracy(top_k=1), 'top@5': Accuracy(top_k=5)})
        self.train_metric = metrics.clone(prefix='train/')
        self.valid_metric = metrics.clone(prefix='valid/')
        self.test_metric = metrics.clone(prefix='test/')

    def forward(self, x):
        return self.fc(self.backbone(x))

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self.shared_step(batch, self.train_metric, 'train', add_dataloader_idx=False)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch, self.valid_metric, 'valid', add_dataloader_idx=True)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch, self.test_metric, 'test', add_dataloader_idx=True)

    def shared_step(self, batch, metric, mode, add_dataloader_idx):
        x, y = batch
        loss, y_hat = self.compute_loss(x, y) if mode == 'train' else self.compute_loss_eval(x, y)
        metric = metric(y_hat, y)
        self.log_dict({f'{mode}/loss': loss}, add_dataloader_idx=add_dataloader_idx)
        self.log_dict(metric, add_dataloader_idx=add_dataloader_idx)
        return loss

    def compute_loss(self, x, y):
        raise NotImplementedError

    def compute_loss_eval(self, x, y):
        y_hat = self.fc(self.backbone(x))
        loss = self.criterion(y_hat, y)
        return loss, y_hat

    def configure_optimizers(self):
        raise NotImplementedError

    def update_and_get_lr_scheduler_config(self):
        if 'num_step' in self.lr_scheduler_init_config['init_args']:
            self.lr_scheduler_init_config['init_args']['num_step'] = self.num_step
        if 'max_epochs' in self.lr_scheduler_init_config['init_args']:
            self.lr_scheduler_init_config['init_args']['max_epochs'] = self.max_epochs
        return self.lr_scheduler_init_config

