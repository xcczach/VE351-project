import numpy as np
from abc import ABC, abstractmethod


class Estimator(ABC):
    @abstractmethod
    def label(self):
        pass

    @abstractmethod
    def estimate_channel(self, Y_k: np.ndarray, X_k: np.ndarray) -> np.ndarray:
        pass


class DirectRatioEstimator(Estimator):
    def label(self):
        return "Direct Ratio Estimator"

    def estimate_channel(self, Y_k: np.ndarray, X_k: np.ndarray) -> np.ndarray:
        ZERO_THRESHOLD = 1e-6
        return np.where(X_k < ZERO_THRESHOLD, 0, Y_k / X_k)


class LeastSquaresEstimator(Estimator):
    def label(self):
        return "Least Squares Estimator"

    def estimate_channel(self, Y_k: np.ndarray, X_k: np.ndarray) -> np.ndarray:
        # Create a diagonal matrix of X_k (nonzero entries only)
        ZERO_THRESHOLD = 1e-6
        nonzero_indices = X_k > ZERO_THRESHOLD  # Identify nonzero X_k
        X_k_filtered = X_k[nonzero_indices]  # Filter nonzero X_k
        Y_k_filtered = Y_k[nonzero_indices]  # Corresponding Y_k values

        # Solve the least squares problem H_k using linear regression
        H_k = np.dot(np.linalg.pinv(np.diag(X_k_filtered)), Y_k_filtered)

        # Return a full-sized result, fill zeros where X_k is zero
        H_k_full = np.zeros_like(X_k, dtype=np.complex_)
        H_k_full[nonzero_indices] = H_k
        return H_k_full


class RegularizedLeastSquaresEstimator(Estimator):
    def __init__(self, lambda_reg: float = 0.1):
        """
        初始化正则化最小二乘估计器。

        参数:
        lambda_reg (float): 正则化参数，用于控制正则化项的强度。
        """
        self.lambda_reg = lambda_reg

    def label(self):
        return f"Regularized Least Squares Estimator (λ={self.lambda_reg})"

    def estimate_channel(self, Y_k: np.ndarray, X_k: np.ndarray) -> np.ndarray:
        """
        使用正则化最小二乘方法估计通道H[k]。

        参数:
        Y_k (np.ndarray): 接收信号，形状为 (N,)。
        X_k (np.ndarray): 发射信号，形状为 (N,)。

        返回:
        np.ndarray: 估计的通道H[k]，形状为 (N,)。
        """
        # 防止除以零，设置阈值
        ZERO_THRESHOLD = 1e-12
        # 计算 |X[k]|^2
        X_abs_sq = np.abs(X_k) ** 2
        # 应用正则化最小二乘公式
        # H[k] = Y[k] * X_conj[k] / (|X[k]|^2 + lambda)
        H_k = np.where(
            X_abs_sq < ZERO_THRESHOLD,
            0,
            Y_k * np.conj(X_k) / (X_abs_sq + self.lambda_reg),
        )
        return H_k
