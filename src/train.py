import torch
from torch import nn
from torch.optim import SGD
import torch.optim.lr_scheduler as LR

from src.model.models import get_model
from src.base_model_wrapper import BaseModelWrapper
from src.cifar import get_cifar, convert_to_dataloader
from src.log import get_log_name, Result


class ModelWrapper(BaseModelWrapper):
    def __init__(self, log_name, model, device, criterion, optimizer):
        super().__init__(log_name)
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer


class MyOpt:
    def __init__(self, model, lr, nbatch, weight_decay=0.0005, momentum=0.95):
        self.optimizer = SGD([
            {'params': model.fc.parameters(), 'lr': lr},
            {'params': [param for name, param in model.named_parameters() if 'fc' not in name], 'lr': lr/10}
        ], lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = LR.MultiStepLR(self.optimizer, milestones=[17, 33], gamma=0.1)
        self.nbatch = nbatch
        self.step_ = 0

    def step(self):
        self.optimizer.step()
        self.step_ += 1
        if self.step_ % self.nbatch == 0:
            self.scheduler.step()
            self.step_ = 0

    def zero_grad(self):
        self.optimizer.zero_grad()


def run(args):
    # step 1. prepare dataset
    train_ds, valid_ds, test_ds = get_cifar(args.dataset, size=args.img_size)
    train_dl, = convert_to_dataloader([train_ds], batch_size=args.batch_size, num_workers=args.num_workers, train=True)
    valid_dl, test_dl = convert_to_dataloader([valid_ds, test_ds], batch_size=args.batch_size,
                                              num_workers=args.num_workers, train=False)

    # step 2. load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model_name, nclass=len(train_ds.classes), pretrained=args.pretrained).to(device)

    # step 3. prepare training tool
    criterion = nn.CrossEntropyLoss()
    optimizer = MyOpt(model=model, nbatch=len(train_dl), lr=args.lr)

    # step 4. train
    model = ModelWrapper(log_name=get_log_name(args), model=model, device=device, optimizer=optimizer,
                         criterion=criterion)
    model.fit(train_dl, valid_dl, test_dl=None, nepoch=args.nepoch)

    # (extra) step 5. save result
    result_saver = Result()
    result_saver.save_result(args, model)