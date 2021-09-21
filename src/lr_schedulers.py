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


class PowerLR(_LRScheduler):
    def __init__(self, optimizer, niter=10000, warmup=500, start_lr=6e-3, last_epoch=-1, verbose=False):
        self.warmup = warmup
        self.start_lr = start_lr
        self.niter = niter
        super(PowerLR, self).__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self, step=None):
        step = step if step else self._step_count
        lrs = []
        for lr in self.base_lrs:
            if step < self.warmup:
                lrs.append(compute_linear(x=step / self.warmup, a=lr - self.start_lr, b=self.start_lr))
            else:
                lrs.append(compute_linear(x=(1 + 10 * (step - self.warmup) / (self.niter - self.warmup)) ** (-0.75), a=lr, b=0))
        return lrs


class FractionLR(_LRScheduler):
    def __init__(self, optimizer, warmup=4000, d_model=512, factor=2, last_epoch=-1, verbose=False):
        self.d_model = d_model
        self.factor = factor
        self.warmup = warmup
        super(FractionLR, self).__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self, step=None):
        step = step if step else self._step_count
        lrs = []
        for lr in self.base_lrs:
            lrs.append(self.factor * (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5)))
        return lrs
