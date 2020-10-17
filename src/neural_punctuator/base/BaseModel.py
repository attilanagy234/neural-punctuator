import logging
import sys
import numpy as np
from abc import abstractmethod
import torch
import torch.nn as nn


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(handler)


class BaseModel(nn.Module):
    """
    Base class for all torch models
    """
    def __init__(self, config):
        super().__init__()
        self._config = config

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def save_model(self, optimizer, epoch):
        log.info("Saving model...")
        torch.save(self.state_dict(), f'saved/models/{self._config.model.name}_{str(epoch)}.pth')
        torch.save(optimizer.state_dict(), f'saved/models/{self._config.model.name}_{str(epoch)}_optimizer_state.pth')

    def load_model(self, optimizer, epoch, model_name):
        log.info("Loading model...")
        self.load_state_dict(torch.load(f'saved/models/{model_name}_{str(epoch)}.pth'))
        optimizer.load_state_dict(torch.load(f'saved/models/{model_name}_{str(epoch)}_optimizer_state.pth'))