from abc import ABC


class BaseWrapper(ABC):
    """
    Base class for wrapper classes
    """
    def __init__(self, config):
        self._config = config
        self._preprocessor = None
        self._classifier = None
        self._trainer = None

    def train(self):
        self._trainer.train()

    def predict(self, *args):
        raise NotImplementedError
