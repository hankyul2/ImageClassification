import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from src.dataset.base_data_module import BaseDataModule
from src.task.base_vision_system import BaseVisionSystem

# parser = argparse.ArgumentParser(description='Computer Vision Image Classification')
# parser.add_argument('-d', '--dataset_name', type=str.lower, default='', choices=[
#     'cifar10', 'cifar100'
# ])
# parser.add_argument('-m', '--model_name', type=str.lower, default='', choices=[
#     'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'wide_resnet50_2',
#     'mobilenet_v2',
#     'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50_32x4d',
#     'vit_base_patch16_224', 'vit_base_patch32_224', 'vit_large_patch16_224', 'vit_large_patch32_224',
#     'vit_base_patch16_384', 'vit_base_patch32_384', 'vit_large_patch16_384', 'vit_large_patch32_384',
#     'r50_vit_base_patch16_224', 'r50_vit_large_patch32_224', 'r50_vit_base_patch16_384', 'r50_vit_large_patch32_384',
# ], help='Enter model name')


if __name__ == '__main__':
    LightningCLI(BaseVisionSystem, BaseDataModule, subclass_mode_data=True, subclass_mode_model=True)
