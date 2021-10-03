from src.cli import MyLightningCLI
from src.data.base_data_module import BaseDataModule
from src.system.base import BaseVisionSystem

if __name__ == '__main__':
    MyLightningCLI(BaseVisionSystem, BaseDataModule, save_config_overwrite=True,
                   subclass_mode_data=True, subclass_mode_model=True)

