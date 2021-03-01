import pandas as pd
from models.MCMC_fourier_windfield import RandomFourierFeatures
from error_estimation.windfield_prediction import get_mean_square_error
import itertools
import numpy as np
import os

import matplotlib.pyplot as plt

# ====================================================================================================================
# DISCLAIMER:
# This code provides an example of how to analyse the data obtained from multiproc_fourier_features_optimisation.py.
# Please make sure to run the aforementioned script at least once before running this code so that you have some data.
# ====================================================================================================================

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir
    path.append(dir(path[0]))


save_folder = "D:/python_runs/windmodelling/Runs/Hyper_optimisation_run/"  # Location of the run

# Define the same grid as in the run.
n_terms = 20
n_steps = [10]
reg_params = np.logspace(-3, 0, 2)
div_params = np.logspace(-3, 0, 2)
gammas = [1.25]
sigmas = [2.25]
params = itertools.product(n_steps, reg_params, div_params, gammas, sigmas)

# Loop through the grid and save the relevant parameters.
error_frame = pd.DataFrame({"burn-in": [], "reg": [], "div": [], "gamma": [], "sigma": [], "NMSE": [], "MSE": []})
for p in params:
    n = p[0]
    r = p[1]
    d = p[2]
    g = p[3]
    s = p[4]

    file_dir = save_folder + f"RandomFeatures/{n}/{r}/{d}/{g}/{s}/" # This string is user specified

    # Loop through and find the data, save in a list
    dir = os.listdir(path=file_dir)
    preds = []
    for file in dir:
        if file[-8:] == ".parquet":
            print("appending file: ", file)
            preds.append(pd.read_parquet(file_dir + file))

    # Calculate the prediction errors for each time snapshot
    print("calculating error: ", file_dir)
    if len(dir) != 0:
        pred = pd.concat(preds)
        norm_error = get_mean_square_error(pred, normalize=True)
        error = get_mean_square_error(pred, normalize=False)
    else:
        norm_error = None
        error = None
    row = pd.DataFrame({"burn-in": [n],
                        "reg": [r],
                        "div": [d],
                        "gamma": [g],
                        "sigma": [s],
                        "NMSE": [norm_error],
                        "MSE": [error]})
    print("done: ", row)
    error_frame = error_frame.append(row)

# The resulting DataFrame can now be used for hyper parameter optimisation:
#                           Simply pick the parameters corresponding to the row with lowest MSE or NMSE.
print(error_frame)