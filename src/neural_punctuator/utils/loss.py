import torch


class WeightedBinaryCrossEntropy:
    def __init__(self, weights=None, reduction='none'):
        self.weights = weights
        self.reduction = reduction

        if self.reduction != 'none':
            raise NotImplementedError('Reduction is not implemented')

    def __call__(self, output, target):
        if self.weights is not None:
            assert len(self.weights) == 2

            loss = self.weights[1] * (target * torch.log(output)) + \
                   self.weights[0] * ((1 - target) * torch.log(1 - output))
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

        return torch.neg(loss)