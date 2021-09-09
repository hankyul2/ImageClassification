from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam
from src.model.models import get_model
from src.base_model_wrapper import BaseModelWrapper
from src.log import Result


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


def remove_log_wrapper(rank):
    def middle_wrapper(c):
        def wrapper(*args, **kwargs):
            instance = c(*args, **kwargs)
            if rank != 0:
                if isinstance(instance, BaseModelWrapper):
                    c.save_best_weight = lambda x, y, w, z: None
                    c.log_tensorboard = lambda a, b, c, d, e, f: None
                    c.log = lambda x, y: None
                elif isinstance(instance, Result):
                    c.save_result = lambda x, y: None
            return instance
        return wrapper
    return middle_wrapper


def apply_wrapper(rank):
    global get_model
    global SGD
    global Adam
    global BaseModelWrapper
    global Result

    get_model = ddp_wrapper(rank)(get_model)
    SGD = shard_optimizer_wrapper(SGD)
    Adam = shard_optimizer_wrapper(Adam)
    BaseModelWrapper = remove_log_wrapper(rank)(BaseModelWrapper)
    Result = remove_log_wrapper(rank)(Result)