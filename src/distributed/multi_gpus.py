import os

import torch.distributed as dist
from torch.multiprocessing import spawn

from src.distributed.dist_wrapper import apply_wrapper
from src.distributed.multi_gpus_tutorial import run_tutorial_DDP_optimizer


def init_process(rank, run, args):
    print('Rank {} init start'.format(rank))
    args.rank = rank
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.rank)
    apply_wrapper(args.rank)
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
    run(args)


def run_multi_gpus(run, args):
    print('Multi GPUs {}'.format(args.is_multi_gpus))
    spawn(fn=init_process, args=(run, args), nprocs=args.world_size)
