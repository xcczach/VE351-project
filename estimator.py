import numpy as np
from abc import ABC, abstractmethod

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
        # Create a diagonal matrix of X_k (nonzero entries only)
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


from scipy.interpolate import interp1d


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
