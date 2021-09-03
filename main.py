import sys
import os
import argparse

import numpy as np
import random

import torch

from src.utils import download_dataset

parser = argparse.ArgumentParser(description='Domain Adaptation')
parser.add_argument('-g', '--gpu_id', type=str, default='', help='Enter which gpu you want to use')
parser.add_argument('-r', '--random_seed', type=int, default=None, help='Enter random seed')
parser.add_argument('-m', '--model_name', type=str.lower, default='', choices=[
    'resnet18', 'resent34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'wide_resnet50_2',
    'vit_base_patch16_224', 'vit_base_patch32_224', 'vit_large_patch16_224', 'vit_large_patch32_224',
    'r50_vit_base_patch16_224', 'r50_vit_large_patch32_224',
], help='Enter model name')
parser.add_argument('-d', '--dataset', type=str, default='', help='Enter dataset')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Enter batch size for train step')
parser.add_argument('-w', '--num_workers', type=int, default=4, help='Enter the number of workers per dataloader')
parser.add_argument('-l', '--lr', type=float, default=3e-4, help='Enter learning rate')
parser.add_argument('-e', '--nepoch', type=int, default=100, help='Enter the number of epoch')
parser.add_argument('-i', '--iter', type=int, default=1, help='Enter the number of iteration you want to run')
parser.add_argument('--data_path', type=str, default='data', help='Enter default data store path')
parser.add_argument('--pretrained', action='store_true', help='If specify, it will use pretrained model')
parser.add_argument('--download_dataset', action='store_true',
                    help='If specify, it will download cifar10/100')


def init(args):
    sys.path.append('.')
    if args.random_seed:
        fix_seed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('DEVICE: {}'.format('cpu' if args.gpu_id == '' else args.gpu_id))


def fix_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)


if __name__ == '__main__':
    args = parser.parse_args()
    init(args)
    print('Model name is {}'.format(args.model_name))

    from src.train import run

    if args.download_dataset:
        download_dataset(args.data_path)
    else:
        for iter in range(args.iter):
            run(args)



