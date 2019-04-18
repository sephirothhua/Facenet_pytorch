# -*- coding: utf-8 -*-
# @Time    : 19-4-18 下午4:35
# @Author  : Altair
# @FileName: w.py
# @Software: PyCharm
# @email   : 641084377@qq.com
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

class ReducePlateauScheduler(_LRScheduler):

    def __init__(self, optimizer, total_epoch):
        self.total_epoch = total_epoch
        super().__init__(optimizer)

    def get_lr(self):
        lr = [0.5 * base_lr * ((np.cos(self.last_epoch/self.total_epoch*np.pi))+1) for base_lr in self.base_lrs]
        return lr

    def step(self, epoch=None):
        return super(ReducePlateauScheduler, self).step(epoch)

def WarmAndReduce_LR(optimizer,base_learning_rate,max_epoch,
                     use_warmup=True,
                     start_learning_rate=1e-5,
                     warmup_epoch=5):
    """ Create an Reduce Learning Rate with or without warm up.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_learning_rate: The basic learning rate ,the same as the regular learning rate.
        max_epoch: The max epoch of training
        use_warmup: Use warm up or not
        start_learning_rate: is active when the use_warmup is True .If active ,it must be the same as the optimizer learning rate.
        warmup_epoch: The epoch to use warm up.
    """
    if(use_warmup):
        ReduceScheduler = ReducePlateauScheduler(optimizer, max_epoch)
        WarmUpScheduler = GradualWarmupScheduler(optimizer, multiplier=int(base_learning_rate / start_learning_rate),
                                        total_epoch=warmup_epoch, after_scheduler=ReduceScheduler)
        return WarmUpScheduler
    else:
        ReduceScheduler = ReducePlateauScheduler(optimizer, max_epoch)
        return ReduceScheduler

if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    lr = []
    base_learning_rate = 0.01
    start_learning_rate = 0.0001
    max_epoch = 100
    warmup_epoch = 1000
    optimizer = torch.optim.SGD([torch.zeros(10)],lr=base_learning_rate)
    WarmUp = WarmAndReduce_LR(optimizer,base_learning_rate,max_epoch,use_warmup=False)
    # ReduceScheduler = ReducePlateauScheduler(optimizer,max_epoch)
    # WarmUp = GradualWarmupScheduler(optimizer,multiplier=int(base_learning_rate/start_learning_rate),total_epoch=1000,after_scheduler=ReduceScheduler)
    for epoch in range(1,100):
        WarmUp.step(epoch)
        lr.append(optimizer.param_groups[0]['lr'])
    plt.plot(np.arange(len(lr)),lr)
    plt.show()
