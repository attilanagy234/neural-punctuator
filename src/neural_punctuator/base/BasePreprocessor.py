from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    """
    Base class for all preprocessors
    """
    def __init__(self, config):
        self._config = config

    @abstractmethod
    def transform(self, *args):
        """
        Transforms:
            - textual descriptions to vectorized features
            - textual output labels to encoded labels
        """
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, *args):
        """
        Transforms encoded labels back to textual labels
        """
        raise NotImplementedError