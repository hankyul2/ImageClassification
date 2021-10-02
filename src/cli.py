import os
from typing import Any

import neptune.new as neptune

from torch.optim import Adam, SGD

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger, CSVLogger
from pytorch_lightning.utilities.cli import LightningCLI

from src.lr_schedulers import CosineLR


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--project_name')
        parser.add_argument('--short_id')
        parser.link_arguments('data.num_classes', 'model.init_args.num_classes', apply_on='instantiate')
        parser.link_arguments('data.num_step', 'model.init_args.num_step', apply_on='instantiate')
        parser.link_arguments('trainer.max_epochs', 'model.init_args.max_epochs', apply_on='parse')
        parser.add_optimizer_args((Adam, SGD), link_to='model.init_args.optimizer_init')
        parser.add_lr_scheduler_args((CosineLR,), link_to='model.init_args.lr_scheduler_init')

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        # 1. load log meta info
        log_dir = 'log'
        config, subcommand = self.get_command_and_config()
        short_id, project_name, dataset_name, model_name = self.get_log_info_from_config(config)
        log_name = self.get_log_name(dataset_name, model_name)

        # 2. define logger
        neptune_logger = self.get_neptune_logger(log_name, project_name, short_id)
        neptune_logger.log_hyperparams(config)
        tensorboard_logger = TensorBoardLogger(log_dir, log_name, neptune_logger.version)
        csv_logger = CSVLogger(log_dir, log_name, neptune_logger.version)
        logger = [neptune_logger, tensorboard_logger, csv_logger]

        # 3. define callback for Checkpoint, LR Scheduler
        save_dir = os.path.join(log_dir, log_name, neptune_logger.version)
        best_save_dir = os.path.join('pretrained', 'in_this_work', log_name)
        callback = [
            ModelCheckpoint(dirpath=save_dir, save_last=True,
                            filename='epoch={epoch}_step={other_metric:.2f}_loss={valid/loss:.3f}',
                            monitor='valid/loss', mode='min', auto_insert_metric_name=False),
            LearningRateMonitor()
        ]

        # 4. pass to trainer
        kwargs = {**kwargs, 'logger': logger, 'default_root_dir': save_dir, 'callbacks': callback}

        return super().instantiate_trainer(**kwargs)

    @staticmethod
    def get_log_name(dataset_name, model_name):
        return f'{model_name}_{dataset_name}'

    @staticmethod
    def get_neptune_logger(log_name, project_name, short_id):
        return NeptuneLogger(
                run=neptune.init(
                    project=project_name,
                    api_token=None,
                    name=log_name,
                    run=short_id,
                ),
            )

    @staticmethod
    def get_log_info_from_config(config):
        short_id = None if config['short_id'] == '' else config['short_id']
        project_name = config['project_name']
        dataset_name = config['data']['init_args']['dataset_name']
        model_name = config['model']['init_args']['backbone_init']['model_name']
        return short_id, project_name, dataset_name, model_name

    def get_command_and_config(self):
        subcommand = self.config['subcommand']
        config = self.config[subcommand]
        return config, subcommand


def get_description():
    model_list = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'wide_resnet50_2',
                  'mobilenet_v2',
                  'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50_32x4d',
                  'vit_base_patch16_224', 'vit_base_patch32_224', 'vit_large_patch16_224', 'vit_large_patch32_224',
                  'vit_base_patch16_384', 'vit_base_patch32_384', 'vit_large_patch16_384', 'vit_large_patch32_384',
                  'r50_vit_base_patch16_224', 'r50_vit_large_patch32_224', 'r50_vit_base_patch16_384',
                  'r50_vit_large_patch32_384']
    dataset_list = ['cifar10', 'cifar100']
    description = "\nmodel list: {}\n dataset_list: {}".format(model_list, dataset_list)
    return description
