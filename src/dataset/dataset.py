from src.dataset.cifar import get_cifar


def get_dataset(dataset_name:str):
    if dataset_name.startswith('cifar'):
        return get_cifar(dataset_name)
