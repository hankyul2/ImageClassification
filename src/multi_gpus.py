import os
from time import sleep

import torch
import torch.distributed as dist
from torch.multiprocessing import spawn


def run_tutorial(args):
    # 1. broadcast
    x = torch.tensor([args.rank], dtype=torch.float)
    print('original tensor: {}'.format(x))
    dist.broadcast(tensor=x, src=0)
    print('broadcast from {} to all tensor({})'.format(0, x))

    # 2. reduce
    x = torch.tensor(args.rank + 1, dtype=torch.float)
    print('original tensor: {}'.format(x))
    dist.reduce(tensor=x, dst=0, op=dist.ReduceOp.SUM)
    print('reduced tensor from rank {} : {}'.format(args.rank, x))

    # 3. all_reduce
    x = torch.tensor(args.rank + 1, dtype=torch.float)
    print('original tensor: {}'.format(x))
    dist.all_reduce(tensor=x, op=dist.ReduceOp.SUM)
    print('all-reduced tensor from rank {} : {}'.format(args.rank, x))

    # 4. scatter
    scatter_list = list(torch.arange(args.world_size).float().squeeze(0)) if args.rank == 1 else None
    x = torch.tensor(0, dtype=torch.float)
    print('original tensor: {}'.format(x))
    dist.scatter(tensor=x, scatter_list=scatter_list, src=1)
    print('scattered tensor from rank {} : {}'.format(args.rank, x))

    # 5. gather
    gather_list = list(torch.zeros((args.world_size, 1)).float()) if args.rank == 1 else None
    x = torch.tensor((args.rank,), dtype=torch.float)
    print('original tensor: {}'.format(x))
    dist.gather(tensor=x, gather_list=gather_list, dst=1)
    print('gathered tensor from rank {} : {}'.format(args.rank, gather_list))

    # 6. all_gather
    gather_list = list(torch.zeros((args.world_size, 1)).float())
    x = torch.tensor((args.rank,), dtype=torch.float)
    print('original tensor: {}'.format(x))
    dist.all_gather(tensor_list=gather_list, tensor=x)
    print('gathered tensor from rank {} : {}'.format(args.rank, gather_list))


def init_process(rank, run, args):
    print('Rank {} init start'.format(rank))
    args.rank = rank
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='gloo', world_size=args.world_size, rank=args.rank)
    run(args)


def run_multi_gpus(run, args):
    print('Multi GPUs {}'.format(args.is_multi_gpus))
    spawn(fn=init_process, args=(run_tutorial, args), nprocs=args.world_size)
