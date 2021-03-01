#!/usr/bin/env python
import random as rand
import pandas as pd
from framework.windfield import Windfield, WindDataFrame
from framework.tools import date_time_to_datetime
from error_estimation.windfield_scoring import score_windfield
import numpy as np
import time
import datetime as dt


def do_many_error_samples(wind_field: Windfield,
                          wind_data: WindDataFrame,
                          k_val_split: int = 5,
                          n_samples: int = 5,
                          seed_no=100) \
                          -> pd.DataFrame:
    """Given a Windfield object and WindData containing measurements from different points in time,
       Sample n_samples of date and time from the WindData and run K-fold crossvalidation on the subset
       of measurements in the Wind data from that specific date and time.

       Parameters
       ----------
       wind_field:  Windfield data object to use in the crossvalidation
       wind_data:   WindDataFrame with data from multiple points in time
                    has to have more than n_samples time points.
       k_val_split: Value to use for K-fold cross validation
       n_samples:   Number of samples to draw from wind_data
       seed_no:     Seed the number generator that draws samples from wind_data.

       Returns
       -------
       error_frame: Dataframe consisting of three rows containing datetime objects
                    corresponding to the times, and errors corresponding to each datetime."""
    rand.seed(seed_no)
    wind_data = date_time_to_datetime(wind_data)
    time_samples = rand.sample(list(wind_data.datetime.unique()), k=n_samples)
    errors = []
    n = 0
    t = time.time()
    for datetime in time_samples:
        sub_frame = wind_data[wind_data.datetime == datetime].reset_index().drop(columns=['index'])
        kfcv_error = score_windfield(wind_field, sub_frame, n_splits=k_val_split, random_state=100)
        errors.append(kfcv_error)
        t_elapsed = time.time() - t
        n += 1
        print('Calculating error at the time: ', pd.to_datetime(datetime).strftime("%d-%b-%Y (%H:%M:%S)"))
        print('{:.3} % done, time remaining:'.format(n / n_samples * 100),
              dt.timedelta(seconds=np.round((n_samples-n)/n*t_elapsed, decimals=0)))
    return pd.DataFrame({'datetime': time_samples, 'error': errors})


def do_bootstrap(error_samples: np.array, n_samples, bootstrap_number=20, seed=100):
    """Sample n_samples errors with replacement a total of bootstrap_number times from error_samples.
        Estimate the variance of the mean of n_samples errors using the bootstrap replicas."""
    rand.seed(seed) # Seed the sampler
    assert n_samples < len(error_samples)
    errors = np.zeros(bootstrap_number)
    # make bootstrap_number bootstraps to estimate variance
    for n in range(bootstrap_number):
        resample = rand.choices(error_samples, k=n_samples)
        errors[n] = np.mean(resample)
    error_mean = np.mean(errors)
    error_var = np.var(errors, ddof=1)
    return error_mean, error_var


def do_many_bootstraps(error_samples: np.array, n_sample_vec, bootstrap_number=20, seed=100):
    """Do the bootstrap for multiple values of n_samples"""
    error_means = []
    error_vars = []
    # For each n in n_sample_vec,
    # estimate the error and it's uncertainty
    for n in n_sample_vec:
        err_mean, err_var = do_bootstrap(error_samples, n_samples=n, bootstrap_number=bootstrap_number, seed=seed)
        error_means.append(err_mean)
        error_vars.append(err_var)
    return error_means, error_vars
