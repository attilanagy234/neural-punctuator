from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    Base class for all trainers
    """
    def __init__(self, model, preprocessor, config):
        self._model = model
        self._preprocessor = preprocessor
        self._config = config


    @abstractmethod
    def train(self, *args):
        """
        Complete training logic
        """
        raise NotImplementedError