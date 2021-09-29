from pytorch_lightning.utilities.cli import LightningCLI

from src.data.base_data_module import BaseDataModule
from src.system.base_vision_system import BaseVisionSystem


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data.num_classes', 'model.init_args.num_classes', apply_on='instantiate')
        parser.link_arguments('data.num_step', 'model.init_args.num_step', apply_on='instantiate')
        parser.link_arguments('trainer.max_epochs', 'model.init_args.max_epochs', apply_on='parse')


if __name__ == '__main__':
    MyLightningCLI(BaseVisionSystem, BaseDataModule, subclass_mode_data=True, subclass_mode_model=True)
