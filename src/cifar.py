import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100


def convert_to_dataloader(datasets, batch_size, num_workers, train=True, sampler_fn=lambda x: None):
    return [torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                                        drop_last=True, sampler=sampler_fn(ds)) for ds in datasets]


def get_cifar(dataset_name, size=(32, 32)):
    train_ds = get_my_dataset(dataset_name, train=True, size=size)
    valid_ds = get_my_dataset(dataset_name, train=False, size=size)
    test_ds = get_my_dataset(dataset_name, train=False, size=size)

    print('{} dataset class num: {}'.format(dataset_name, len(train_ds.classes)))
    print('{} train dataset len: {}'.format(dataset_name, len(train_ds)))
    print('{} valid dataset len: {}'.format(dataset_name, len(valid_ds)))
    print('{} test dataset len: {}'.format(dataset_name, len(test_ds)))

    return train_ds, valid_ds, test_ds


def get_my_dataset(dataset_name, train, size):
    if dataset_name == 'cifar10':
        dataset = CIFAR10(root='data', train=train, download=False)
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset_name == 'cifar100':
        dataset = CIFAR100(root='data', train=train, download=False)
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

    my_dataset = MyDataset(ds=dataset, transforms=transforms_fn)

    return my_dataset


class MyDataset(object):
    def __init__(self, ds, transforms=None):
        super().__init__()
        self.ds = ds
        self.transforms = transforms
        self.classes = ds.classes
        self.class_to_idx = ds.class_to_idx

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]

        if self.transforms:
            img = self.transforms(img)

        return img, label