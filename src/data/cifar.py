import random
from time import sleep

from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from src.data.base_data_module import BaseDataModule


class CIFAR(BaseDataModule):
    def __init__(self, dataset_name: str, size: tuple, **kwargs):
        if dataset_name == 'cifar10':
            dataset, mean, std = CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        elif dataset_name == 'cifar100':
            dataset, mean, std = CIFAR100, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)

        train_transform, test_transform = self.get_trasnforms(mean, std, size)
        super(CIFAR, self).__init__(dataset_name, dataset, train_transform, test_transform, **kwargs)

    def get_trasnforms(self, mean, std, size):
        train = transforms.Compose([
            transforms.Resize(size),
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return train, test


class Noisy_CIFAR(CIFAR):
    def __init__(self, *args, noisy_ratio: float = 0.3, **kwargs):
        super(Noisy_CIFAR, self).__init__(*args, **kwargs)
        self.noisy_ratio = noisy_ratio

    def setup(self, stage: str = None):
        if stage in (None, 'fit'):
            ds = self.dataset(root=self.data_root, train=True, transform=self.train_transform, target_transform=self.generate_noisy_label)
            self.train_ds, self.valid_ds = self.split_train_valid(ds)

        elif stage in (None, 'test', 'predict'):
            self.test_ds = self.dataset(root=self.data_root, train=False, transform=self.test_transform)

    def generate_noisy_label(self, x):
        return x if random.random() > self.noisy_ratio else random.randint(0, self.num_classes - 1)


class Small_CIFAR(CIFAR):
    def __init__(self, *args, sample_num: int = 10, **kwargs):
        super(Small_CIFAR, self).__init__(*args, **kwargs)
        self.sample_num = sample_num

    def setup(self, stage: str = None):
        if stage in (None, 'fit'):
            ds = self.dataset(root=self.data_root, train=True, transform=self.train_transform)
            self.train_ds, self.valid_ds, _ = self.split_train_valid(ds)

        elif stage in (None, 'test', 'predict'):
            self.test_ds = self.dataset(root=self.data_root, train=False, transform=self.test_transform)

    def split_train_valid(self, ds):
        ds_len = self.sample_num * self.num_classes
        valid_ds_len = int(ds_len * self.valid_ratio)
        train_ds_len = ds_len - valid_ds_len
        return random_split(ds, [train_ds_len, valid_ds_len, len(ds) - ds_len])