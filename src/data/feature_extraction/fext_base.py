"""Base class for Feature extractors"""
from abc import ABC, abstractmethod
from numpy import ndarray


SUPPORTED_BACKENDS = ["sklearn", "keras"]
JOINED_BACKENDS = ", ".join(SUPPORTED_BACKENDS)
SUPPORTED_DCLASS = [ndarray]
JOINED_DCLASS = ", ".join([cls.__name__ for cls in SUPPORTED_DCLASS])


class FeatureExtractor(ABC):
    def __init__(self, backend: str = "sklearn"):
        self.backend = backend

    @property    
    def backend(self):
        return self._backend
    @backend.setter
    def backend(self, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(
                "backend must be a string, a type '{}' was given!".
                format(type(value).__name__)
            )
        if value not in SUPPORTED_BACKENDS:
            raise TypeError(
                "backend must be one of {}, {} was given!".
                format(JOINED_BACKENDS, type(value).__name__)
            )
        self._backend = value

    def transform(self, feature_data: ndarray) -> ndarray:
        shape = feature_data.shape
        if self.backend == "sklearn" and len(shape) >= 3:
            return feature_data.reshape(shape[0], shape[1]*shape[2])
        return feature_data

    @abstractmethod
    def extract(self, windowed_data: ndarray) -> ndarray:
        pass
