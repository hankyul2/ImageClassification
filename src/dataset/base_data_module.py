from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torchvision import transforms


class BaseDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_name: str,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 size: tuple = (224, 224),
                 data_root: str = 'data',
                 valid_ratio: float = 0.1):
        """
        Base Data Module

        :param dataset_name: Enter which dataset name
        :param batch_size: Enter batch size
        :param num_workers: Enter number of workers
        :param size: Enter resized image
        :param data_root: Enter root data folder name
        :param valid_ratio: Enter valid dataset ratio
        """
        super(BaseDataModule, self).__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.data_root = data_root
        self.valid_ratio = valid_ratio
        self.mean = None
        self.std = None
        self.Dataset = None
        self.train_transform = None
        self.test_transform = None

    def prepare_data(self) -> None:
        train = self.Dataset(root=self.data_root, train=True, download=True)
        test = self.Dataset(root=self.data_root, train=True, download=True)

        print('-' * 50)
        print('* {} dataset class num: {}'.format(self.dataset_name, len(train.classes)))
        print('* {} train dataset len: {}'.format(self.dataset_name, len(train)))
        print('* {} test dataset len: {}'.format(self.dataset_name, len(test)))
        print('-' * 50)

    def setup(self, stage: str = None):
        if stage in (None, 'fit'):
            ds = self.Dataset(root=self.data_root, train=True, transform=transforms.ToTensor())
            self.split_train_valid(ds)

        elif stage in (None, 'test'):
            self.test_ds = self.Dataset(root=self.data_root, train=False, transform=transforms.ToTensor())

    def split_train_valid(self, ds):
        ds_len = len(ds)
        valid_ds_len = int(ds_len * self.valid_ratio)
        train_ds_len = ds_len - valid_ds_len
        self.train_ds, self.valid_ds = random_split(ds, [train_ds_len, valid_ds_len])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
