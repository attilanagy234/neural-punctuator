from abc import ABC, abstractmethod


class BaseTuner(ABC):
    """
    Base class for tuner classes
    """
    @abstractmethod
    def tune(self, *args):
        raise NotImplementedError
