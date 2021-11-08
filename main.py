import os
import warnings

warnings.filterwarnings('ignore')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from src.cli import MyLightningCLI
from src.data.base_data_module import BaseDataModule
from src.system.base import BaseVisionSystem

if __name__ == '__main__':
    cli = MyLightningCLI(BaseVisionSystem, BaseDataModule, save_config_overwrite=True,
                   subclass_mode_data=True, subclass_mode_model=True)
    cli.trainer.test(ckpt_path='best')

