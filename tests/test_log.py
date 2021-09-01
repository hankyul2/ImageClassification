import shutil
from pathlib import Path
from time import sleep

import pytest
import torch
from easydict import EasyDict as edict

from src.log import Result, get_current_time


def test_setup_directory():
    result = Result()
    assert Path('log').exists() == True
    assert Path('result/best_result').exists() == True


def test_setup_logfile():
    result = Result()
    assert Path('log/result.csv').exists() == True


def test_arg2result():
    result = Result()
    args = edict({
        'model_name': 'resnet50',
        'dataset': 'cifar10',
        'nepoch': 200,
        'lr': 0.001,
        'batch_size': 128
    })
    model = edict({
        'start_time': '2021-09-01/10:06:40',
        'best_acc': torch.tensor(50.0).item(),
        'best_epoch': 110,
        'log_best_weight_path': 'log/best_weight/...'
    })
    result.arg2result(args, model)


def test_get_no():
    shutil.rmtree('log/')
    result = Result()
    assert result.get_no() == 1


def test_save_result():
    args = edict({
        'model_name': 'resnet50',
        'dataset': 'cifar10',
        'nepoch': 200,
        'lr': 0.001,
        'batch_size': 128
    })
    model = edict({
        'start_time': '2021-09-01/10:06:40',
        'best_acc': torch.tensor(50.0).item(),
        'best_epoch': 110,
        'log_best_weight_path': 'log/best_weight/...'
    })
    result = Result()
    result.save_result(args, model)
    assert result.get_no() == 2


def test_update_best_result():
    args = edict({
        'model_name': 'resnet50',
        'dataset': 'cifar10',
        'nepoch': 200,
        'lr': 0.001,
        'batch_size': 128
    })
    model = edict({
        'start_time': '2021-09-01/10:06:40',
        'best_acc': torch.tensor(50.0).item(),
        'best_epoch': 110,
        'log_best_weight_path': 'log/best_weight/...'
    })
    result = Result()
    result.save_result(args, model)
    result.update_best_result()
    assert Path('result/best_result/resnet50.md').exists() == True


def test_get_identity_columns():
    result_list = [
        ['1', 'resnet50', 'cifar10', ''],
        ['2', 'resnet110', 'cifar10', ''],
        ['3', 'resnet50', 'cifar100', ''],
        ['4', 'resnet110', 'cifar100', ''],
        ['5', 'resnet110', 'cifar100', ''],
    ]
    result = Result()
    assert len(result.get_identity_columns(result_list)) == 4
    assert len(result.get_identity_columns(result_list, 'resnet50')) == 2


def test_get_same_type():
    args = edict({
        'model_name': 'resnet50',
        'dataset': 'cifar10',
        'nepoch': 200,
        'lr': 0.001,
        'batch_size': 128
    })
    model = edict({
        'start_time': '2021-09-01/10:06:40',
        'best_acc': torch.tensor(50.0).item(),
        'best_epoch': 110,
        'log_best_weight_path': 'log/best_weight/...'
    })
    result = Result()
    result.save_result(args, model)
    identity_column_name = ['resnet50', 'cifar10']
    identity_column_idx = [1, 2]
    assert len(result.get_same_list(identity_column_name, identity_column_idx)) > 0


def test_filter_best_result():
    result = Result()
    result_list = [
        ['1', 'resnet50', 'cifar10', '', '98.0'],
        ['2', 'resnet110', 'cifar10', '', '98.1'],
        ['3', 'resnet50', 'cifar100', '', '98.2'],
        ['4', 'resnet110', 'cifar100', '', '98.3'],
        ['5', 'resnet110', 'cifar100', '', '98.4'],
    ]
    assert len(result.filter_best_result(result_list, 1)) == 1
    assert len(result.filter_best_result(result_list, 2)) == 2
    assert len(result.filter_best_result(result_list, 3)) == 3


def test_make_log_readme():
    model_name = 'resnet110'
    logs = [[['hello', 'bye', 'bye', 'bye', '0.1', '', '', '', '', '', ''],
             ['hello', 'bye', 'bye', 'bye', '0.1', '', '', '', '', '', '']]]
    result = Result()
    result.make_log_readme(model_name, logs)
    assert Path('result/best_result/resnet110.md').exists() == True


def test_get_current_time():
    time_str = get_current_time()
    assert time_str
    assert isinstance(time_str, str)


def test_get_best_pretrained_model_path():
    args = edict({
        'model_name': 'resnet50',
        'dataset': 'cifar10',
        'nepoch': 200,
        'lr': 0.001,
        'batch_size': 128
    })
    model = edict({
        'start_time': '2021-09-01/10:06:40',
        'best_acc': torch.tensor(50.0).item(),
        'best_epoch': 110,
        'log_best_weight_path': 'log/best_weight/...'
    })
    result = Result()
    result.save_result(args, model)
    result.get_best_pretrained_model_path('resnet50', 'cifar10')
