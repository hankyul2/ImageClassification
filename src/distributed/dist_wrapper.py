from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam
from src.model.models import get_model


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


def apply_wrapper(rank):
    global get_model
    global SGD
    global Adam

    get_model = ddp_wrapper(rank)(get_model)
    SGD = shard_optimizer_wrapper(SGD)
    Adam = shard_optimizer_wrapper(Adam)