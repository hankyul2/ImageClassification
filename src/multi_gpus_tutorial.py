"""
This is from pytorch DDP tutorial. If you want to follow this up, I recommend to do basic steps that I did.
1. Study "Distributed PyTorch" for understanding how basic distributed works
2. Study "PyTorch로 분산 어플리케이션 개발하기" for understanding how collective communication works.
    - you have to understand torch.nn.parallel.DistributedDataParallel class
    - you have to understand torch.multiprocessing.spawn(fn, args, nprocs) for basic multiprocessing in pytorch
    - you have to understand torch.distributed.init_process_group(backend, init_method, rank, world_size)
    - you have to understand torch.distributed.scatter, gather, all_gather, reduce, all_reduce, broadcast
    - you have to understand how backend and init_method works
3. Study "https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html" for Shard Optimizer.
    - you have to understand how torch.distributed.ZeroRedundancyOptimizer works.
4. Study Python Wrapper, which makes code very simple.
"""
import torch
from torch import distributed as dist, nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.optim import SGD


def run_tutorial_collective_communication(args):
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


def optimizer_wrapper(optimizer_class):
    def wrapper(*args, **kwargs):
        optimizer = ZeroRedundancyOptimizer(*args, optimizer_class=optimizer_class, **kwargs)
        return optimizer
    return wrapper


def run_tutorial_DDP_optimizer(args):
    global SGD

    device = torch.device('cuda:{}'.format(args.rank) if torch.cuda.is_available() else 'cpu')

    x = torch.rand(100, 50).to(device)
    y = torch.randint(10, size=(100,)).to(device)
    model = nn.Sequential(nn.Linear(50, 2000), nn.Linear(2000, 2000), nn.Linear(2000, 10)).to(device)
    model = DDP(model, device_ids=[args.rank])

    SGD = optimizer_wrapper(SGD)
    optimizer = SGD(model.parameters(), lr=0.01, weight_decay=1e-5)

    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    loss.backward()

    optimizer.step()

    print("Max Memory Usage: {}MB".format(torch.cuda.max_memory_allocated(args.rank) // 1e6))