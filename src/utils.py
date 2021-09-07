import glob
import os
import subprocess
from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100
from torch.hub import load_state_dict_from_url
import torch

import numpy as np


model_urls = {
    # ResNet
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',

    # ViT
    'vit_base_patch16_224': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
    'vit_base_patch32_224': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz',
    'vit_large_patch16_224': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz',
    'vit_large_patch32_224': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz',

    # Hybrid (resnet50 + ViT)
    'r50_vit_base_patch16_224': 'https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz',
    'r50_vit_large_patch32_224': 'https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-L_32.npz',
}


def load_from_zoo(model, model_name, pretrained_path='pretrained_path'):
    model_name = change_384_224(model_name)
    Path(os.path.join(pretrained_path, model_name)).mkdir(parents=True, exist_ok=True)
    if model_urls[model_name].endswith('pth'):
        state_dict = load_state_dict_from_url(url=model_urls[model_name],
                                              model_dir=os.path.join(pretrained_path, model_name),
                                              progress=True, map_location='cpu')
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)
    elif model_urls[model_name].endswith('npz'):
        npz = load_npz_from_url(url=model_urls[model_name],
                                file_name=os.path.join(pretrained_path, model_name, os.path.basename(model_urls[model_name])))
        model.load_npz(npz)


def change_384_224(model_name):
    model_name = model_name.replace('384', '224')
    return model_name


def load_npz_from_url(url, file_name):
    subprocess.run(["wget", "-r", "-nc", '-O', file_name, url])
    return np.load(file_name)


def download_dataset(data_path):
    Path(data_path).mkdir(exist_ok=True, parents=True)
    CIFAR10(root=data_path, download=True)
    CIFAR100(root=data_path, download=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
