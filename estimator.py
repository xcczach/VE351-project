import numpy as np
from abc import ABC, abstractmethod


class Estimator(ABC):
    @abstractmethod
    def estimate_channel(self, Y_k: np.ndarray, X_k: np.ndarray) -> np.ndarray:
        pass


class DirectRatioEstimator(Estimator):
    def estimate_channel(self, Y_k: np.ndarray, X_k: np.ndarray) -> np.ndarray:
        ZERO_THRESHOLD = 1e-20
        return np.where(X_k < ZERO_THRESHOLD, 0, Y_k / X_k)