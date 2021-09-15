import torch
from torch import nn
from torch.optim import SGD
import torch.optim.lr_scheduler as LR

from src.lr_schedulers import CosineLR
from src.model.models import get_model
from src.base_model_wrapper import BaseModelWrapper
from src.cifar import get_cifar, convert_to_dataloader
from src.log import Result


class ModelWrapper(BaseModelWrapper):
    def __init__(self, log_name, start_time, model, device, criterion, optimizer):
        super().__init__(log_name, start_time)
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer


class MyOpt:
    def __init__(self, model, lr, niter, weight_decay=0.0005, momentum=0.95):
        self.optimizer = SGD([
            {'params': list(set(param for name, param in model.named_parameters() if 'fc' in name)), 'factor': 1},
            {'params': list(set(param for name, param in model.named_parameters() if 'fc' not in name)), 'factor': 0.1}
        ], lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = CosineLR(self.optimizer, niter, warmup=500, lr=lr)

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


def run(args):
    # step 1. prepare dataset
    train_ds, valid_ds, test_ds = get_cifar(args.dataset, size=args.img_size)
    train_dl, = convert_to_dataloader([train_ds], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valid_dl, test_dl = convert_to_dataloader([valid_ds, test_ds], batch_size=args.batch_size,
                                              num_workers=args.num_workers, shuffle=False)

    # step 2. load model
    device = torch.device('cuda:{}'.format(args.rank) if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model_name, nclass=len(train_ds.dataset.classes), device=device,
                      pretrained=args.pretrained, dropout=args.dropout)

    # step 3. prepare training tool
    criterion = nn.CrossEntropyLoss()
    optimizer = MyOpt(model=model, niter=len(train_dl)*args.nepoch, lr=args.lr)

    # step 4. train
    model = ModelWrapper(log_name=args.log_name, start_time=args.start_time, model=model, device=device, optimizer=optimizer,
                         criterion=criterion)
    model.fit(train_dl, valid_dl, nepoch=args.nepoch)

    # step 5. evaluate
    model.load_best_weight()
    model.evaluate(test_dl)

    # (extra) step 6. save result
    result_saver = Result()
    result_saver.save_result(args, model)