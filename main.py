from pytorch_lightning.utilities.cli import LightningCLI

from src.data.base_data_module import BaseDataModule
from src.system.base_vision_system import BaseVisionSystem


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--data_list', action='store_true', help="""[
            'cifar10', 'cifar100',
        ]""")
        parser.add_argument('--model_list', action='store_true', help="""[
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'wide_resnet50_2',
            'mobilenet_v2',
            'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50_32x4d',
            'vit_base_patch16_224', 'vit_base_patch32_224', 'vit_large_patch16_224', 'vit_large_patch32_224',
            'vit_base_patch16_384', 'vit_base_patch32_384', 'vit_large_patch16_384', 'vit_large_patch32_384',
            'r50_vit_base_patch16_224', 'r50_vit_large_patch32_224', 'r50_vit_base_patch16_384', 'r50_vit_large_patch32_384',
        ]""")
        parser.link_arguments('data.num_classes', 'model.init_args.num_classes', apply_on='instantiate')
        parser.link_arguments('data.num_step', 'model.init_args.num_step', apply_on='instantiate')
        parser.link_arguments('trainer.max_epochs', 'model.init_args.max_epochs', apply_on='parse')

if __name__ == '__main__':
    MyLightningCLI(BaseVisionSystem, BaseDataModule, subclass_mode_data=True, subclass_mode_model=True)
