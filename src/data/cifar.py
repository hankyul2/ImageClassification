from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from src.data.base_data_module import BaseDataModule


def get_cifar_img_transform(mean, std, size, mode):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(size),
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    elif mode == 'test':
        return transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


class CIFAR10_DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super(CIFAR10_DataModule, self).__init__(*args, **kwargs)
        self.Dataset = CIFAR10
        self.num_classes = 10
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)
        self.train_transform = get_cifar_img_transform(self.mean, self.std, self.size, 'train')
        self.test_transform = get_cifar_img_transform(self.mean, self.std, self.size, 'test')
        self.prepare_data()


class CIFAR100_DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super(CIFAR100_DataModule, self).__init__(*args, **kwargs)
        self.Dataset = CIFAR100
        self.num_classes = 100
        self.mean = (0.5071, 0.4865, 0.4409)
        self.std = (0.2673, 0.2564, 0.2762)
        self.train_transform = get_cifar_img_transform(self.mean, self.std, self.size, 'train')
        self.test_transform = get_cifar_img_transform(self.mean, self.std, self.size, 'test')
        self.prepare_data()

