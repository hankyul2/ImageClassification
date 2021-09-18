import math


def compute_linear(x, a, b):
    return x * a + b


class CosineLR:
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
            cosine_y = (1 + math.cos(math.pi * (step - self.warmup) / (self.niter - self.warmup))) / 2
            return compute_linear(x=cosine_y, a=self.lr, b=0)


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