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
    'vit_base_patch16_384', 'vit_base_patch32_384', 'vit_large_patch16_384', 'vit_large_patch32_384',
    'r50_vit_base_patch16_224', 'r50_vit_large_patch32_224', 'r50_vit_base_patch16_384', 'r50_vit_large_patch32_384',
], help='Enter model name')
parser.add_argument('-d', '--dataset', type=str, default='', help='Enter dataset')
parser.add_argument('-s', '--img_size', type=int, default=(224, 224), nargs='+', help='Enter Image Size')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Enter batch size for train step')
parser.add_argument('-w', '--num_workers', type=int, default=4, help='Enter the number of workers per dataloader')
parser.add_argument('-l', '--lr', type=float, default=3e-4, help='Enter learning rate')
parser.add_argument('-e', '--nepoch', type=int, default=50, help='Enter the number of epoch')
parser.add_argument('-i', '--iter', type=int, default=1, help='Enter the number of iteration you want to run')
parser.add_argument('--data_path', type=str, default='data', help='Enter default data store path')
parser.add_argument('--pretrained', action='store_true', help='If specify, it will use pretrained model')
parser.add_argument('--download_dataset', action='store_true',
                    help='If specify, it will download cifar10/100')
parser.add_argument('--update_best_result', action='store_true', help='If specify, it will update best result log')


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
    print('image size is {}'.format(args.img_size))

    if args.download_dataset:
        download_dataset(args.data_path)
        exit(0)
    elif args.update_best_result:
        from src.log import run
    else:
        from src.train import run

    for iter in range(args.iter):
        run(args)



