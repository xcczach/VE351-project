import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d

ZERO_THRESHOLD = 1e-6


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
        return np.where(X_k < ZERO_THRESHOLD, 0, Y_k / X_k)


class LeastSquaresEstimator(Estimator):
    def label(self):
        return "Least Squares Estimator"

    def estimate_channel(self, Y_k: np.ndarray, X_k: np.ndarray) -> np.ndarray:
        nonzero_indices = X_k > ZERO_THRESHOLD
        X_k_filtered = X_k[nonzero_indices]
        Y_k_filtered = Y_k[nonzero_indices]

        H_k = np.dot(np.linalg.pinv(np.diag(X_k_filtered)), Y_k_filtered)

        H_k_full = np.zeros_like(X_k, dtype=np.complex_)
        H_k_full[nonzero_indices] = H_k
        return H_k_full


class RegularizedLeastSquaresEstimator(Estimator):
    def __init__(self, lambda_reg: float = 0.1):
        self.lambda_reg = lambda_reg

    def label(self):
        return f"Regularized Least Squares Estimator (λ={self.lambda_reg})"

    def estimate_channel(self, Y_k: np.ndarray, X_k: np.ndarray) -> np.ndarray:
        X_abs_sq = np.abs(X_k) ** 2
        H_k = np.where(
            X_abs_sq < ZERO_THRESHOLD,
            0,
            Y_k * np.conj(X_k) / (X_abs_sq + self.lambda_reg),
        )
        return H_k


class MMSEEstimator(Estimator):
    def __init__(self, var_H2: float, var_W2: float):
        self.var_H2 = var_H2
        self.var_W2 = var_W2

    def label(self):
        return f"MMSE Estimator (σ_H²={self.var_H2}, σ_W²={self.var_W2})"

    def estimate_channel(self, Y_k: np.ndarray, X_k: np.ndarray) -> np.ndarray:
        X_abs_sq = np.abs(X_k) ** 2
        coeff = self.var_H2 * X_abs_sq / (X_abs_sq * self.var_H2 + self.var_W2)
        H_k = coeff * (Y_k * np.conj(X_k)) / (X_abs_sq + 1e-12)
        return H_k


class PolynomialInterpolationEstimator(RegularizedLeastSquaresEstimator):

    def __init__(self, lambda_reg: float, degree: int):
        super().__init__(lambda_reg=lambda_reg)
        self.degree = degree

    def label(self):
        return f"Polynomial Interpolation Estimator (degree={self.degree}, λ={self.lambda_reg})"

    def estimate_channel(self, Y_k, X_k):
        H_k = super().estimate_channel(Y_k, X_k)

        known_indices = np.where(np.abs(H_k) > ZERO_THRESHOLD)[0]
        unknown_indices = np.where(np.abs(H_k) < ZERO_THRESHOLD)[0]

        if len(known_indices) < self.degree + 1:
            return H_k

        H_known = H_k[known_indices]

        poly_real = np.polyfit(known_indices, H_known.real, self.degree)
        poly_imag = np.polyfit(known_indices, H_known.imag, self.degree)

        poly_real_fn = np.poly1d(poly_real)
        poly_imag_fn = np.poly1d(poly_imag)

        H_est_real = poly_real_fn(unknown_indices)
        H_est_imag = poly_imag_fn(unknown_indices)
        H_est = H_est_real + 1j * H_est_imag
        H_k[unknown_indices] = H_est

        return H_k


class SplineInterpolationEstimator(RegularizedLeastSquaresEstimator):
    def __init__(self, lambda_reg: float, kind="cubic"):
        super().__init__(lambda_reg=lambda_reg)
        self.kind = kind

    def label(self):
        return f"Spline Interpolation Estimator (λ={self.lambda_reg}, kind={self.kind})"

    def estimate_channel(self, Y_k, X_k):
        H_k = super().estimate_channel(Y_k, X_k)

        known_indices = np.where(np.abs(H_k) > ZERO_THRESHOLD)[0]
        unknown_indices = np.where(np.abs(H_k) < ZERO_THRESHOLD)[0]

        if len(known_indices) < 2:
            return H_k

        H_known = H_k[known_indices]

        spline_real = interp1d(
            known_indices,
            H_known.real,
            kind="cubic",
            fill_value="extrapolate",
        )
        spline_imag = interp1d(
            known_indices,
            H_known.imag,
            kind="cubic",
            fill_value="extrapolate",
        )

        H_est_real = spline_real(unknown_indices)
        H_est_imag = spline_imag(unknown_indices)
        H_est = H_est_real + 1j * H_est_imag
        H_k[unknown_indices] = H_est

        return H_k


class KalmanFilterEstimator(Estimator):
    def __init__(
        self,
        Q: float = 1e-5,
        R: float = 1e-2,
    ):
        self.Q = Q
        self.R = R

    def label(self):
        return f"Kalman Filter Estimator (Q={self.Q}, R={self.R})"

    def estimate_channel(self, Y_k: np.ndarray, X_k: np.ndarray) -> np.ndarray:
        H_new = np.zeros_like(X_k, dtype=np.complex_)
        num_subcarriers = len(X_k)
        H_est = np.zeros(num_subcarriers, dtype=np.complex_)
        P = np.ones(num_subcarriers)

        for k in range(num_subcarriers):
            if np.abs(X_k[k]) < ZERO_THRESHOLD:
                H_new[k] = H_est[k]
                continue

            H_pred = H_est[k]
            P_pred = P[k] + self.Q

            X_conj = np.conj(X_k[k])
            denominator = (np.abs(X_k[k]) ** 2) * P_pred + self.R
            K = P_pred * X_conj / denominator

            innovation = Y_k[k] - X_k[k] * H_pred
            H_updated = H_pred + K * innovation

            P_updated = (1 - K * X_k[k]) * P_pred

            H_new[k] = H_updated
            H_est[k] = H_updated
            P[k] = P_updated

        return H_new
