import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


class Simulator:
    def __init__(self, N: int, Ts: float):
        self.N = N
        self.Ts = Ts
        self.t = np.arange(0, N * Ts, Ts)
        self.f = fftfreq(N, Ts)

    def create_channel_ht(self, amplitudes: list[float], delays: list[float]):
        if len(amplitudes) != len(delays):
            raise ValueError("amplitudes and delays must have the same length")
        h_t = np.zeros(self.N)
        for amp, delay in zip(amplitudes, delays):
            idx = np.argmin(np.abs(self.t - delay))
            h_t[idx] = amp
        return h_t

    def fft(self, x_t: np.ndarray):
        return fft(x_t)

    def create_pilot_Xk(self, p: int):
        X_k = np.zeros(self.N)
        X_k[::p] = 1
        return X_k

    def create_gaussian_noise_Wk(
        self, SNR: float, p: int, H_k: np.ndarray, X_k: np.ndarray
    ):
        SNR = 10 ** (SNR / 10)
        P_noise = p * np.sum(np.abs(H_k * X_k) ** 2) / (self.N * SNR)
        return np.sqrt(P_noise) * np.random.randn(self.N)

    def create_Y_k(self, H_k: np.ndarray, X_k: np.ndarray, W_k: np.ndarray):
        return H_k * X_k + W_k

    def mse(self, H_k_hat: np.ndarray, H_k: np.ndarray):
        return np.mean(np.abs(H_k_hat - H_k) ** 2)

    def freq_mag_plot(self, signal_K: np.ndarray, plot_filename: str):
        """
        For real-valued time-domain signals only.
        """
        normalized_signal_K = 2.0 / self.N * np.abs(signal_K[0 : self.N // 2])
        normalized_signal_K[0] /= 2
        plt.figure()
        plt.plot(self.f[0 : self.N // 2], normalized_signal_K)
        plt.grid()
        plt.xlabel("Freq (Hz)")
        plt.ylabel("Magnitude (normalized)")
        plt.title("Frequency Magnitude Plot")
        plt.savefig(plot_filename)

    def comparison_freq_mag_plot(
        self, signal_K1: np.ndarray, signal_K2: np.ndarray, plot_filename: str
    ):
        """
        For real-valued time-domain signals only.
        """
        normalized_signal_K1 = 2.0 / self.N * np.abs(signal_K1[0 : self.N // 2])
        normalized_signal_K1[0] /= 2
        normalized_signal_K2 = 2.0 / self.N * np.abs(signal_K2[0 : self.N // 2])
        normalized_signal_K2[0] /= 2
        plt.figure()
        plt.plot(self.f[0 : self.N // 2], normalized_signal_K1, label="True")
        plt.plot(
            self.f[0 : self.N // 2],
            normalized_signal_K2,
            label="Estimated",
            linestyle="--",
        )
        plt.grid()
        plt.xlabel("Freq (Hz)")
        plt.ylabel("Magnitude (normalized)")
        plt.title(f"Frequency Magnitude Plot, filename={plot_filename}")
        plt.legend()
        plt.savefig(plot_filename)


if __name__ == "__main__":
    N = 1024
    Ts = 1e-4
    p = 8
    SNR = 10
    sim = Simulator(N, Ts)
    h_t = sim.create_channel_ht([13, 10], [0, 5 * sim.Ts])
    X_k = sim.create_pilot_Xk(p)
    H_k = sim.fft(h_t)
    W_k = sim.create_gaussian_noise_Wk(SNR, p, H_k, X_k)
    Y_k = sim.create_Y_k(H_k, X_k, W_k)
    sim.freq_mag_plot_for_real_signal(H_k, "H_k.png")
    print(h_t.shape, X_k.shape, H_k.shape, W_k.shape, Y_k.shape)
