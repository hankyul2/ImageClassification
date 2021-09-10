import os
import sys

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam
from src.model.models import get_model
from src.base_model_wrapper import BaseModelWrapper
from src.log import Result
from src.cifar import convert_to_dataloader


def dataloader_wrapper(fn):
    def wrpper(*args, **kwargs):
        out = fn(*args, **kwargs, sampler_fn=torch.utils.data.distributed.DistributedSampler)
        return out
    return wrpper

def ddp_wrapper(rank):
    def middle_wrapper(fn):
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            out = DDP(out, device_ids=[rank])
            return out

        return wrapper

    return middle_wrapper


def shard_optimizer_wrapper(optimizer_class):
    def wrapper(*args, **kwargs):
        optimizer = ZeroRedundancyOptimizer(*args, optimizer_class=optimizer_class, **kwargs)
        return optimizer

    return wrapper


def remove_log_wrapper(rank, log_name, start_time):
    def middle_wrapper(c):
        if rank != 0:
            c.__init__ = lambda *args, **kwargs: None
            c.save_best_weight = lambda x, y, w, z: None
            c.log_tensorboard = lambda a, b, c, d, e, f: None
            c.log = lambda x, y: None
            c.trace_handler = torch.profiler.tensorboard_trace_handler('log/tensor_board/{}/{}'.format(
                log_name, start_time
            ))
            c.save_result = lambda x, y: None
        return c

    return middle_wrapper


def ignore_stdout(rank):
    if rank != 0:
        f = open(os.devnull, 'w')
        sys.stdout = f
        sys.stderr = f


def apply_wrapper(rank, log_name, start_time):
    global convert_to_dataloader
    global get_model
    global SGD
    global Adam
    global BaseModelWrapper
    global Result

    convert_to_dataloader = dataloader_wrapper(convert_to_dataloader)
    get_model = ddp_wrapper(rank)(get_model)
    SGD = shard_optimizer_wrapper(SGD)
    Adam = shard_optimizer_wrapper(Adam)
    BaseModelWrapper = remove_log_wrapper(rank, log_name, start_time)(BaseModelWrapper)
    Result = remove_log_wrapper(rank, log_name, start_time)(Result)
    ignore_stdout(rank)
