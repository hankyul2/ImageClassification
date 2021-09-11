import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam
from src.model.models import get_model
from src.base_model_wrapper import BaseModelWrapper
from src.log import Result
from src.cifar import convert_to_dataloader
from src.utils import AverageMeter


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


def barrier_wrapper(rank):
    def middle_wrapper(fn):
        def wrapper(*args, **kwargs):
            dist.barrier(device_ids=[rank])
            out = fn(*args, **kwargs)
            return out

        return wrapper

    return middle_wrapper


def metric_wrapper(world_size):
    def middle_wrapper(fn):
        def wrapper(self, val, n=1):
            fn(self, val, n=n)
            if torch.is_tensor(val):
                dist.reduce(self.avg, dst=0, op=dist.ReduceOp.SUM)
                self.avg /= world_size

        return wrapper

    return middle_wrapper


def model_wrapper(rank, log_name, start_time):
    def middle_wrapper(c):
        c.load_best_weight = barrier_wrapper(rank)(c.load_best_weight)
        if rank != 0:
            c.__init__ = lambda *args, **kwargs: None
            c.log = lambda x, y: None
            c.log_tensorboard = lambda a, b, c, d, e, f: None
            c.save_best_weight = lambda x, y, w, z: None
            c.log_best_weight_path = 'log/best_weight/{}/{}.pth'.format(log_name, start_time)
            c.trace_handler = torch.profiler.tensorboard_trace_handler(
                'log/tensor_board/{}/{}'.format(log_name, start_time)
            )
        return c

    return middle_wrapper


def result_wrapper(rank):
    def middle_wrapper(c):
        if rank != 0:
            c.save_result = lambda x, y, w: None
        return c

    return middle_wrapper


def ignore_stdout(rank):
    if rank != 0:
        f = open(os.devnull, 'w')
        sys.stdout = f
        sys.stderr = f


def apply_wrapper(rank, world_size, log_name, start_time):
    global convert_to_dataloader
    global get_model
    global SGD
    global Adam
    global AverageMeter
    global BaseModelWrapper
    global Result

    convert_to_dataloader = dataloader_wrapper(convert_to_dataloader)
    get_model = ddp_wrapper(rank)(get_model)
    SGD = shard_optimizer_wrapper(SGD)
    Adam = shard_optimizer_wrapper(Adam)
    AverageMeter.update = metric_wrapper(world_size)(AverageMeter.update)
    BaseModelWrapper = model_wrapper(rank, log_name, start_time)(BaseModelWrapper)
    Result = result_wrapper(rank)(Result)
    ignore_stdout(rank)
