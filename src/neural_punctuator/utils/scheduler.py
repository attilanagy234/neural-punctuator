from torch.optim.lr_scheduler import _LRScheduler


class LinearScheduler(_LRScheduler):

    def __init__(self, optimizer, max_steps=10000):
        self.max_steps = max_steps
        self.lr = 0
        super(LinearScheduler, self).__init__(optimizer, -1)

    def get_lr(self):
        self.lr = self.base_lrs[-1] * min(1, self._step_count/self.max_steps)
        return [base_lr * min(1, self._step_count/self.max_steps)
                for base_lr in self.base_lrs]