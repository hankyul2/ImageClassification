from typing import Any

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.utilities.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--project_name')
        parser.link_arguments('data.num_classes', 'model.init_args.num_classes', apply_on='instantiate')
        parser.link_arguments('data.num_step', 'model.init_args.num_step', apply_on='instantiate')
        parser.link_arguments('trainer.max_epochs', 'model.init_args.max_epochs', apply_on='parse')

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        config = self.config[self.config['subcommand']]
        project_name = config['project_name']
        dataset_name = config['data']['init_args']['dataset_name']
        model_name = config['model']['init_args']['backbone_init']['model_name']

        kwargs = {**kwargs, 'logger':[
            NeptuneLogger(
                api_key=None,
                project=project_name,
                name=f'{model_name}_{dataset_name}',
                tags=[model_name, dataset_name]
            )
        ]}
        return super().instantiate_trainer(**kwargs)


def get_description():
    model_list = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'wide_resnet50_2',
            'mobilenet_v2',
            'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50_32x4d',
            'vit_base_patch16_224', 'vit_base_patch32_224', 'vit_large_patch16_224', 'vit_large_patch32_224',
            'vit_base_patch16_384', 'vit_base_patch32_384', 'vit_large_patch16_384', 'vit_large_patch32_384',
            'r50_vit_base_patch16_224', 'r50_vit_large_patch32_224', 'r50_vit_base_patch16_384', 'r50_vit_large_patch32_384']
    dataset_list = ['cifar10', 'cifar100']
    description = "\nmodel list: {}\n dataset_list: {}".format(model_list, dataset_list)
    return description