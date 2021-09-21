import math

from torch.optim.lr_scheduler import _LRScheduler


def compute_linear(x, a, b):
    return x * a + b


class CosineLR(_LRScheduler):
    def __init__(self, optimizer, niter=10000, warmup=500, start_lr=6e-3, last_epoch=-1, verbose=False):
        self.niter = niter
        self.warmup = warmup
        self.start_lr = start_lr
        super(CosineLR, self).__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self, step=None):
        step = step if step else self._step_count
        lrs = []
        for lr in self.base_lrs:
            if step < self.warmup:
                lrs.append(compute_linear(x=step / self.warmup, a=lr - self.start_lr, b=self.start_lr))
            else:
                cosine_y = (1 + math.cos(math.pi * (step - self.warmup) / (self.niter - self.warmup))) / 2
                lrs.append(compute_linear(x=cosine_y, a=lr, b=0))
        return lrs


# class CosineLR:
#     def __init__(self, optimizer, niter=10000, warmup=500, lr=0.03, start_lr=6e-3):
#         self.optimizer = optimizer
#         self.niter = niter
#         self.warmup = warmup
#         self.lr = lr
#         self.start_lr = start_lr
#         self.step_ = 0
#
#     def step(self):
#         self.step_ += 1
#         lr = self.get_lr()
#         for param in self.optimizer.param_groups:
#             param['lr'] = lr
#
#     def get_lr(self, step=None):
#         step = step if step else self.step_
#         if step < self.warmup:
#             return compute_linear(x=step / self.warmup, a=self.lr - self.start_lr, b=self.start_lr)
#         else:
#             cosine_y = (1 + math.cos(math.pi * (step - self.warmup) / (self.niter - self.warmup))) / 2
#             return compute_linear(x=cosine_y, a=self.lr, b=0)
#
#     def state_dict(self):
#         return {'lr': self.lr, 'step_': self.step_, 'start_lr': self.start_lr,
#                 'warmup': self.warmup,'niter': self.niter}
#
#     def load_state_dict(self, state_dict):
#         self.__dict__.update(state_dict)


class PowerLR:
    def __init__(self, optimizer, niter=10000, warmup=500, lr=0.03, start_lr=6e-3):
        self.optimizer = optimizer
        self.niter = niter
        self.warmup = warmup
        self.lr = lr
        self.start_lr = start_lr
        self.step_ = 0

    def step(self):
        self.step_ += 1
        lr = self.get_lr()
        for param in self.optimizer.param_groups:
            param['lr'] = lr

    def get_lr(self, step=None):
        step = step if step else self.step_
        if step < self.warmup:
            return compute_linear(x=step / self.warmup, a=self.lr - self.start_lr, b=self.start_lr)
        else:
            return compute_linear(x=(1 + 10 * (step - self.warmup) / (self.niter - self.warmup)) ** (-0.75), a=self.lr,
                                  b=0)


class FractionLR:
    def __init__(self, optimizer, warmup=4000, d_model=512, factor=2):
        self.optimizer = optimizer
        self.d_model = d_model
        self.factor = factor
        self.warmup = warmup
        self.step_ = 0

    def step(self):
        self.step_ += 1
        rate = self.get_lr()
        for param in self.optimizer.param_groups:
            param['lr'] = rate

    def get_lr(self, step=None):
        step = step if step else self.step_
        return self.factor * (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5))
