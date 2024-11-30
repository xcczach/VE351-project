import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.fft import ifft

ZERO_THRESHOLD = 1e-6


class Estimator(ABC):
    @abstractmethod
    def label(self):
        pass

    @abstractmethod
    def estimate_channel(self, Y_k: np.ndarray, X_k: np.ndarray) -> np.ndarray:
        """
        Estimate the channel response H_k from the received signal Y_k and the transmitted signal X_k.
        Y_k and X_k are both real-valued vectors of the same length N, in frequency domain, shape (N,).
        """
        pass


class DirectRatioEstimator(Estimator):
    def label(self):
        return "Direct Ratio Estimator (baseline)"

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


class PolynomialInterpolationEstimator(Estimator):

    def __init__(self, degree: int):
        self.degree = degree

    def label(self):
        return f"Polynomial Interpolation Estimator (degree={self.degree})"

    def estimate_channel(self, Y_k, X_k):
        H_k = DirectRatioEstimator().estimate_channel(Y_k, X_k)

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


class SplineInterpolationEstimator(Estimator):
    def __init__(self, kind="cubic"):
        self.kind = kind

    def label(self):
        return f"Spline Interpolation Estimator (kind={self.kind})"

    def estimate_channel(self, Y_k, X_k):
        H_k = DirectRatioEstimator().estimate_channel(Y_k, X_k)

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


class CompressedSensingEstimator(Estimator):
    def label(self):
        return "Compressed Sensing Channel Estimator (OMP)"

    def estimate_channel(self, Y_k: np.ndarray, X_k: np.ndarray) -> np.ndarray:
        """
        使用压缩感知（OMP）算法估计OFDM信道

        参数:
        - Y_k: 接收的导频符号向量，形状为 (N,)
        - X_k: 发送的导频符号向量，形状为 (N,)

        返回:
        - H_hat: 估计的频域信道响应向量，形状与 Y_k 和 X_k 相同 (N,)
        """
        # 确定子载波数量 N
        N = len(X_k)

        # 确定导频位置 (导频符号为1的位置)
        pilot_positions = np.where(X_k != 0)[0]
        M = len(pilot_positions)

        # 提取接收的导频符号并进行归一化
        Y_p = Y_k[pilot_positions]
        X_p = X_k[pilot_positions]

        # 避免除以零
        if np.any(X_p == 0):
            raise ValueError("0 in X_p")

        y = Y_p / X_p  # 形状为 (M,)

        # 构建部分离散傅里叶变换（DFT）矩阵 F_p
        F = np.fft.fft(np.eye(N)) / np.sqrt(N)  # 标准化的DFT矩阵，形状为 (N, N)
        F_p = F[pilot_positions, :]  # 选择导频子载波对应的行，形状为 (M, N)

        # 假设信道在时域中是稀疏的，使用单位基矩阵作为稀疏基矩阵 Psi
        Psi = np.eye(N, dtype=complex)  # 形状为 (N, N)

        # 构建测量矩阵 Phi = F_p * Psi = F_p
        Phi = F_p @ Psi  # 形状为 (M, N)

        # 定义稀疏度 K
        K = 20  # 可以根据实际情况调整

        # 使用OMP算法分别恢复实部和虚部
        # 由于 scikit-learn 的 OMP 不支持复数，分开处理实部和虚部
        omp_real = OrthogonalMatchingPursuit(n_nonzero_coefs=K)
        omp_real.fit(Phi.real, y.real)
        h_real = omp_real.coef_  # 形状为 (N,)

        omp_imag = OrthogonalMatchingPursuit(n_nonzero_coefs=K)
        omp_imag.fit(
            Phi.real, y.imag
        )  # 这里只使用 Phi.real 来拟合虚部，可以根据需要调整
        h_imag = omp_imag.coef_  # 形状为 (N,)

        # 组合实部和虚部得到时域信道 h_hat
        h_hat = h_real + 1j * h_imag  # 形状为 (N,)

        # 将时域信道转换回频域信道 H_hat = FFT(h_hat)
        H_hat_full = np.fft.fft(h_hat, n=N)  # 形状为 (N,)

        # 返回完整频域信道 H_hat
        return H_hat_full


class MUSICEstimator(Estimator):
    def label(self):
        return "MUSIC Estimator (Single Antenna)"

    def estimate_channel(
        self, Y_k: np.ndarray, X_k: np.ndarray, num_sources: int = 2
    ) -> np.ndarray:
        """
        Estimate the channel response using MUSIC algorithm.

        Parameters:
        - Y_k: Received signal array of shape (N,), in the frequency domain.
        - X_k: Transmitted signal array of shape (N,), in the frequency domain.
        - num_sources: Number of dominant signal sources (e.g., multi-path components).

        Returns:
        - H_est: Estimated channel response of shape (N,).
        """
        N = len(Y_k)

        # Step 1: Construct observed signal Z_k
        Z_k = Y_k / np.where(X_k < ZERO_THRESHOLD, 1, X_k)  # Avoid division by zero

        # Step 2: Construct covariance matrix
        # Since input is 1D, use Toeplitz approximation to estimate covariance
        R = np.correlate(Z_k, Z_k, mode="full")[
            N - 1 :
        ]  # Autocorrelation (simplified covariance)
        R_matrix = np.zeros((N, N), dtype=np.complex128)
        for i in range(N):
            R_matrix[i, : N - i] = R[: N - i]  # Fill covariance matrix row by row
            R_matrix[: N - i, i] = np.conjugate(R[: N - i])  # Fill symmetric parts

        # Step 3: Perform eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(R_matrix)
        idx = eigenvalues.argsort()[::-1]  # Sort eigenvalues in descending order
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Step 4: Separate signal subspace and noise subspace
        U_n = eigenvectors[:, num_sources:]  # Noise subspace

        # Step 5: Compute MUSIC pseudo-spectrum
        music_spectrum = np.zeros(N, dtype=np.float64)
        for k in range(N):
            steering_vector = self._steering_vector(k, N)
            music_spectrum[k] = 1 / np.linalg.norm(np.dot(U_n.T, steering_vector)) ** 2

        # Step 6: Identify peaks in the MUSIC spectrum
        peaks = np.argpartition(music_spectrum, -num_sources)[-num_sources:]

        # Step 7: Construct the estimated channel response
        H_est = np.zeros(N, dtype=np.complex128)
        for peak in peaks:
            H_est[peak] = music_spectrum[peak]

        return H_est

    def _steering_vector(self, k, N):
        """
        Generate a steering vector for the given frequency index.

        Parameters:
        - k: Frequency index.
        - N: Total number of subcarriers.

        Returns:
        - Steering vector of shape (N, 1).
        """
        return np.exp(1j * 2 * np.pi * k * np.arange(N) / N).reshape(-1, 1)
