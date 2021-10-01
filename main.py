from src.cli import MyLightningCLI, get_description
from src.data.base_data_module import BaseDataModule
from src.system.base import BaseVisionSystem

if __name__ == '__main__':
    MyLightningCLI(BaseVisionSystem, BaseDataModule,
                   subclass_mode_data=True, subclass_mode_model=True, description=get_description())
