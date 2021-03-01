import pandas as pd
from models.MCMC_fourier_windfield import RandomFourierFeatures
# from models.quadratic_fourier_windfield import FourierWindfield
from models.MCMC_fourier_windfield import RandomFourierFeatures as MeanFWF
from models.averaging_windfield import AveragingWindfield
from error_estimation.windfield_prediction import get_mean_square_error
import itertools
import numpy as np
from pathos.multiprocessing import freeze_support
import os
import matplotlib.pyplot as plt

if __name__ == "__main__" and __package__ is None:
    freeze_support()
    from sys import path
    from os.path import dirname as dir
    path.append(dir(path[0]))

# Define multidim grid of parameters
# Third run: MCMC reg, n_steps optimisation
data_folder = "D:/python_runs/windmodelling/Data/"
save_folder = "D:/python_runs/windmodelling/Runs/burnin_run"
n_terms = 400
n_steps = 500
reg_param = 0.017
div_param = 0.001
gamma = 1.4
sigma = 2.5

# grid_search = pd.DataFrame({"reg": [], "div": [], "NMSE": [], "MSE": []})
def get_pred():
    wf = RandomFourierFeatures(n_terms=n_terms,
                               n_steps=n_steps,
                               reg_param=reg_param,
                               div_param=div_param,
                               gamma=gamma,
                               sigma=sigma,
                               seed=100)

    file_dir = save_folder + wf.to_string()
    preds = []
    dir = os.listdir(path=file_dir)
    for file in dir:
        if file[-8:] == ".parquet":
            print("appending file: ", file)
            preds.append(pd.read_parquet(file_dir + file))

    print("calculating error: ", file_dir)
    assert len(dir) > 0, "No runs have been made"

    pred = pd.concat(preds).drop("index", axis=1).reset_index(drop=True)
    return pred

pred = get_pred()
pred2 = pred[[d[0:7] != "2018-09" for d in pred["date"]]]
pred2.to_parquet("emastr_analysis/Scripts/2020-07-08-No_storm_errors/mcmcwf_preds_optimal.parquet")



get_mean_square_error(pred2, normalize=True)


# plot_burnin(error_frame, n_steps)
#plot_sigma_gamma(error_frame, gammas, sigmas)