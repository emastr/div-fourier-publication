import pandas as pd
from models.MCMC_fourier_windfield import RandomFourierFeatures
# from models.quadratic_fourier_windfield import FourierWindfield
from models.averaging_windfield import AveragingWindfield
from error_estimation.windfield_prediction import do_parallel_5fold, split_and_save_data
from framework.data_loader import load_parquet
import itertools
import numpy as np
from pathos.multiprocessing import freeze_support
from sys import path
from os.path import dirname as dir


# Prevent matrix inversion from using multiprocessing,
# Only parallelize the prediction process based on the time slices.
let_numpy_use_multiproc = False
if not let_numpy_use_multiproc:
    import os
    os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6


if __name__ == "__main__" and __package__ is None:
    freeze_support()
    path.append(dir(path[0]))

    # Folder to put the data in
    data_folder = "D:/python_runs/windmodelling/Data/"

    # Folder to save the runs in
    save_folder = "D:/python_runs/windmodelling/Runs/Hyper_optimisation_run/"

    # Load data
    data = load_parquet("Data/wind_data_2018.parquet")
    data = data[data['date'].apply(lambda f: f[0:7]) != '2018-09']      # Remove high outlier data from a storm in september.

    # Divide into time slices and save locally for reference later
    #split_and_save_data(data, data_folder)      # Comment this out if you already ran the code once

    # Specify parameter grid.
    # Make sure to copy the grid setup as the filing system does not explicitely keep track of this for you
    n_terms = 20
    n_steps = [10]
    reg_params = np.logspace(-3, 0, 2)
    div_params = np.logspace(-3, 0, 2)
    gammas = [1.25]
    sigmas = [2.25]
    params = itertools.product(n_steps, reg_params, div_params, gammas, sigmas)


    # Generate windfields corresponding to the parameters
    # Generate save directories for each windfield.
    # Preallocate lists
    windfields = []
    save_folders = []
    for p in params:
        n = p[0]
        r = p[1]
        d = p[2]
        g = p[3]
        s = p[4]
        windfields.append(RandomFourierFeatures(n_terms=n_terms, n_steps=n, reg_param=r, div_param=d, gamma=g, sigma=s, seed=100))
        save_folders.append(save_folder + f"RandomFeatures/{n}/{r}/{d}/{g}/{s}/")

    do_parallel_5fold(windfields, data_folder, save_folders, processes=7)

# ============ DISCLAIMER ==============================================================
# The data is now ready for post processing, see multiproc_fourier_features_analysis.py.
