from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from src.dataset.base_data_module import BaseDataModule


class CIFAR10_DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super(CIFAR10_DataModule, self).__init__(*args, **kwargs)
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)
        self.Dataset = CIFAR10
        self.train_transforms = transforms.Compose([
            transforms.Resize(self.size),
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(self.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])


class CIFAR100_DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super(CIFAR100_DataModule, self).__init__(*args, **kwargs)
        self.mean = (0.5071, 0.4865, 0.4409)
        self.std = (0.2673, 0.2564, 0.2762)
        self.Dataset = CIFAR100
        self.train_transforms = transforms.Compose([
            transforms.Resize(self.size),
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(self.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

