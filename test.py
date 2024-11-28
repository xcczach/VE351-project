from simulator import Simulator
from estimator import DirectRatioEstimator

if __name__ == "__main__":
    N = 1024
    Ts = 1e-4
    simulator = Simulator(N, Ts)
    p_arr = [8, 16]
    SNR_arr = [0, 5, 10, 15, 20, 25, 30]
    h_t_amplitudes = [[13, 10, 7, 5, 3], [13, 10, 7, 5, 3]]
    h_t_shifts = [
        [0, 5 * Ts, 12 * Ts, 30 * Ts, 50 * Ts],
        [0, 5.4 * Ts, 12.5 * Ts, 30 * Ts, 50.5 * Ts],
    ]
    for p in p_arr:
        X_k = simulator.create_pilot_Xk(p)
        for h_t_ind in range(len(h_t_amplitudes)):
            h_t = simulator.create_channel_ht(
                h_t_amplitudes[h_t_ind], h_t_shifts[h_t_ind]
            )
            H_k = simulator.fft(h_t)
            for SNR in SNR_arr:
                W_k = simulator.create_gaussian_noise_Wk(SNR, p, H_k, X_k)
                Y_k = simulator.create_Y_k(H_k, X_k, W_k)
                estimator = DirectRatioEstimator()
                H_k_hat = estimator.estimate_channel(Y_k, X_k)
                mse = simulator.mse(H_k_hat, H_k)
                print(f"p={p}, SNR={SNR}, mse={mse}")
                simulator.comparison_freq_mag_plot(H_k, H_k_hat, f"H_k_{p}_{SNR}.png")
