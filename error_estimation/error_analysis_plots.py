#!/usr/bin/env python
import framework.data_loader as dl
import pandas as pd
from error_estimation.error_uncertainty import do_many_bootstraps
from error_estimation.windfield_prediction import get_square_error
from error_estimation.windfield_prediction import get_mean_square_error
import random as rand
from models.nn_windfield import NearestNeighbourWindfield
import framework.plotter as pl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Histogram:

    def __init__(self, data, bins):
        d_max = max(data)
        d_min = min(data)
        width = (d_max-d_min)/bins
        edges = [d_min + n*width for n in range(1, bins+1)] # Right edges of bins
        freqs = np.zeros(bins)                              # frequencies

        for d in data:
            if d == d_max:
                freqs[-1] += 1
            else:
                bin_number = int(np.floor((d-d_min)/(d_max-d_min)*bins))
                freqs[bin_number] += 1

        freqs /= sum(freqs*width)

        self.edges = edges
        self.freqs = freqs
        self.cumfr = np.cumsum(freqs*width)

    def get_r2(self, cdf):
        delta = self.cumfr - cdf(self.edges)
        return 1 - np.mean(delta**2) / np.var(self.cumfr)


def plot_ridgeline(windfield_errors):
    """
    Ridgline plot of prediction errors for a set of models.

    :param windfield_errors: pandas DataFrame with errors. Each column corresponds to one model.
            Each entry should contain the mean square error for a fixed time sample, the mean taken over all stations:

            Ex: windfield errors =

                 index  | model 1  | model 2  |  ... | model n |
                 =================================================
                 time 1 |  mse 11  |  mse 12  |  ... |  mse 1n |
                 time 2 |  mse 21  |  mse 22  |  ... |  mse 2n |
                 ...
                 time m |  mse m1  |  mse m2  |  ... |  mse mn |
                 =================================================
    """

    models = windfield_errors.columns # Extract model names
    n_points = 30                     # Number of bins for the ridge plot

    # Get mean and confidence intervals for each model,
    # Also create histogram for each models
    model_data = pd.DataFrame()
    for model in models:
        data = {}
        error = windfield_errors[model].dropna().values
        data["name"] = model
        data["mse"] = np.mean(error)
        data["mse_ci"] = 2 * np.std(error) / len(error) ** 0.5
        data["hist"] = Histogram(error, n_points)
        model_data = model_data.append(data, ignore_index=True)

    # Sort values by mean square error to make the plot more visually appealing
    model_data = model_data.sort_values(by=["mse"], axis=0).reset_index(drop=True)

    y = 0    # First histogram vertical position
    ys = []  # List of vertical histogram positions
    error_grid = np.linspace(0, 2, n_points)

    # Create colors for the histograms
    cmap = cm.get_cmap("bwr", len(models))
    colors = [cmap(i) for i in range(len(models))]

    plt.figure(figsize=(6, 4))
    ax_L = plt.gca()
    for i, data in model_data.iterrows():
        ys.append(y)
        e_mean = data["mse"]
        e_error = data["mse_ci"]
        hist = data["hist"]

        freqs = hist.freqs
        bins = hist.edges

        freqs = np.array([0] + [a for a in freqs for _ in range(2)] + [0])
        width = bins[1] - bins[0]
        bins = np.array([bins[0]] + [a + da for a in bins for da in [0, width]] + [bins[-1]+width])

        ax_L.fill_between(bins, y * np.ones(2*n_points+2), y + freqs, alpha=0.5, color=colors[i])
        ax_L.plot(bins, freqs + y, color='k', linewidth=1)

        def get_y(x):
            return freqs[2 * int((x - min(bins)) / (max(bins) - min(bins)) * n_points) + 2]

        freq_mean = get_y(e_mean)
        freq_lower = get_y(e_mean - e_error)
        freq_upper = get_y(e_mean + e_error)

        ax_L.plot([e_mean, e_mean], [y, freq_mean + y], color='k')
        ax_L.plot([e_mean-e_error, e_mean-e_error], [y, freq_lower + y], "k--", linewidth=.5)
        ax_L.plot([e_mean+e_error, e_mean+e_error], [y, freq_upper + y], "k--", linewidth=.5)

        y += 1.1*max(freqs)

    ax_L.plot(error_grid, np.zeros(n_points), color='k', linewidth=2)
    ax_L.set_yticks(ys)
    ax_L.set_ylim([0, y])
    ax_L.set_yticklabels([])

    ax_L.set_yticklabels(model_data["name"])

    ax_L.set_xlabel("Fraction of unexplained variance $\widetilde{\mathcal{E}}$")
    ax_L.set_xlim([0, 1.2])
    ax_L.grid(axis="y")
    plt.tight_layout()
    return None


def get_error_table(errors, mean_square_wind, print_tex_string=False):
    """
    Get a table of results given
    :param errors: pandas DataFrame with errors. Each column corresponds to one model.
            Each entry should contain the mean square error for a fixed time sample, the mean taken over all stations:

            Ex: Windfield errors =

                 index  | model 1  | model 2  |  ... | model n |
                 =================================================
                 time 1 |  mse 11  |  mse 12  |  ... |  mse 1n |
                 time 2 |  mse 21  |  mse 22  |  ... |  mse 2n |
                 ...
                 time m |  mse m1  |  mse m2  |  ... |  mse mn |
                 =================================================
    :param mean_square_wind: Mean square wind speed of the data involved in the errors DataFrame.
    :return: Pandas DataFrame containing relative and absolute errors as well as confidence intervals:

                | name    | nMSE | nMSEci | MSE | MSEci |
                | model 1 |      |        |     |       |
                | model 2 |      |        |     |       |
                 ...
                | model n |      |        |     |       |

            where,

            nMSE:   Normalised Mean square error
            nMSEci: Confidence intervals for nMSE.
            MSE:    Mean square error
            MSEci:  Confidence intervals for MSE.
    """
    table = pd.DataFrame()
    u = mean_square_wind
    for model in errors.columns:
        mean = errors[model].mean()
        std = errors[model].std()/len(errors.index)**0.5
        #table = \
        #    table.append({"name": model, "nMSE": mean, "nMSEci": 2*std, "MSE": mean*u, "MSEci": 2*std*u}, ignore_index=True)
        table = \
            table.append({"name": model, "nMSE": mean, "nMSEci": 2 * std, "MSE": mean * u, "MSEci": 2 * std * u},
                         ignore_index=True)

    table = table.sort_values("nMSE", ascending=False)
    if print_tex_string:
        for _, row in table.iterrows():
            model = row["name"]
            nmean = row["nMSE"]
            nstd = row["nMSEci"]
            mean = row["MSE"]
            std = row["MSEci"]
            print(f"{model} & {nmean:.3f} $\pm$ {nstd:.3f} & {mean:.3f} $\pm$ {std:.3f} \\\\")
    return table


def clt_plot(error: pd.Series, n_max=800):
    """Make plots showing bootstrapped error std against
        estimations made using the central limit theorem"""
    plt.figure()
    n_values = [n for n in range(20, n_max, 20)]
    error_mean, error_var = do_many_bootstraps(error, n_values, bootstrap_number=50)
    plt.plot(n_values, np.sqrt(error_var), label='Bootstrap estimate')
    plt.plot(n_values, np.divide(np.std(error, ddof=1), np.sqrt(n_values)), label='CLT estimate')
    plt.xlabel('number of samples')
    plt.ylabel('variance of mean square error')
    plt.legend()
    plt.show()

