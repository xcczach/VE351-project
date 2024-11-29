from simulator import Simulator
from estimator import (
    Estimator,
    DirectRatioEstimator,
    LeastSquaresEstimator,
    RegularizedLeastSquaresEstimator,
    PolynomialInterpolationEstimator,
    SplineInterpolationEstimator,
)
from typing import List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    estimators: List[Estimator] = [
        DirectRatioEstimator(),
        LeastSquaresEstimator(),
        RegularizedLeastSquaresEstimator(),
        PolynomialInterpolationEstimator(lambda_reg=0.1, degree=5),
        SplineInterpolationEstimator(lambda_reg=0.1, kind="cubic"),
    ]
    results = []
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
                for estimator in estimators:
                    H_k_hat = estimator.estimate_channel(Y_k, X_k)
                    mse = simulator.mse(H_k_hat, H_k)
                    results.append(
                        {
                            "Estimator": estimator.label(),
                            "ht": h_t_ind,
                            "p": p,
                            "SNR": SNR,
                            "MSE": mse,
                        }
                    )
                    simulator.comparison_freq_mag_plot(
                        H_k,
                        H_k_hat,
                        f"plots/Hk_{h_t_ind}_{estimator.label()}_{p}_{SNR}_mse{mse.round()}.png",
                    )
    # Create a DataFrame from the results and plot the MSE vs SNR
    df = pd.DataFrame(results)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    line_styles = {8: "solid", 16: "dashed"}
    line_widths = {0: 1.0, 1: 2.0}
    estimators_labels = df["Estimator"].unique()
    p_values = df["p"].unique()
    ht_values = df["ht"].unique()
    palette = sns.color_palette("tab10", n_colors=len(estimators_labels))
    estimator_colors = {
        estimator: color for estimator, color in zip(estimators_labels, palette)
    }
    for estimator in estimators_labels:
        for p in p_values:
            for ht in ht_values:
                subset = df[
                    (df["Estimator"] == estimator) & (df["p"] == p) & (df["ht"] == ht)
                ]
                subset = subset.sort_values("SNR")
                # Only label curves with p=8 and ht=0 for the legend
                if p == 8 and ht == 0:
                    label = f"{estimator}, p={p}, ht={ht}"
                else:
                    label = None
                ax.plot(
                    subset["SNR"],
                    subset["MSE"],
                    label=label,
                    linestyle=line_styles[p],
                    linewidth=line_widths[ht],
                    color=estimator_colors[estimator],
                    marker="o",
                )
    legend_elements = []
    for estimator in estimators_labels:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=estimator_colors[estimator],
                linestyle=line_styles[8],
                linewidth=line_widths[0],
                marker="o",
                label=f"{estimator}",
            )
        )
    legend_elements_p = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=line_styles[p],
            linewidth=2,
            label=f"p={p}",
        )
        for p in p_values
    ]
    legend_elements_ht = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=line_widths[ht],
            label=f"ht={ht}",
        )
        for ht in ht_values
    ]

    legend_estimators = ax.legend(
        handles=legend_elements,
        title="Estimator, p=8, ht=0",
        fontsize="small",
        ncol=2,
        bbox_to_anchor=(1, 1),
        loc="upper right",
    )
    legend_p = ax.legend(
        handles=legend_elements_p,
        title="p",
        fontsize="small",
        ncol=2,
        bbox_to_anchor=(1, 0.9),
        loc="upper right",
    )
    ax.legend(
        handles=legend_elements_ht,
        title="ht",
        fontsize="small",
        ncol=2,
        bbox_to_anchor=(1, 0.8),
        loc="upper right",
    )
    ax.add_artist(legend_estimators)
    ax.add_artist(legend_p)

    ax.set_title("MSE vs SNR for Different Estimators, p, and ht Values")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plots/MSE_vs_SNR.png")
