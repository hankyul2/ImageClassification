import torch
from torch.utils.data import random_split, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100


def convert_to_dataloader(datasets, batch_size, num_workers, shuffle=True, sampler_fn=lambda x: None):
    return [torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                        drop_last=True, sampler=sampler_fn(ds)) for ds in datasets]


def get_cifar(dataset_name, size=(32, 32)):
    train_org_ds = get_cifar_dataset(dataset_name, train=True, size=size)
    test_ds = get_cifar_dataset(dataset_name, train=False, size=size)
    train_ds, valid_ds = split_train_valid(train_org_ds)

    print('{} dataset class num: {}'.format(dataset_name, len(train_org_ds.classes)))
    print('{} train dataset len: {}'.format(dataset_name, len(train_ds)))
    print('{} valid dataset len: {}'.format(dataset_name, len(valid_ds)))
    print('{} test dataset len: {}'.format(dataset_name, len(test_ds)))

    return train_ds, valid_ds, test_ds


def split_train_valid(org_ds, valid_ratio=0.1, seed=1997):
    return random_split(org_ds, [int(len(org_ds) * (1 - valid_ratio)), int(len(org_ds) * valid_ratio)],
                        generator=torch.Generator().manual_seed(seed))


def get_cifar_dataset(dataset_name, train, size, root='data'):
    if dataset_name == 'cifar10':
        dataset = CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset_name == 'cifar100':
        dataset = CIFAR100
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)

    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    transforms_fn = train_transform if train else test_transform

    cifar_dataset = dataset(root=root, train=train, download=False, transform=transforms_fn)

    return cifar_dataset