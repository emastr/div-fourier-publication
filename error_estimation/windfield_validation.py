import importlib
from framework.windfield import Windfield, WindDataFrame
from error_estimation.windfield_scoring import score_windfield
from error_estimation.windfield_scoring import average_square_error
import itertools
from scipy import stats
import random
import numpy as np
from datetime import date
import pandas as pd
from models.zero_windfield import ZeroWindfield

from sklearn.model_selection import KFold
from framework.windfield import Windfield, WindDataFrame
import numpy as np
import time

def filter_times(wind_data, m):
    """ Utility function for 'get_dates()'. Return a DataFrame of  where measurements times for which there
    are less than m stations have been removed."""
    time_samples = wind_data[['date', 'time']]
    cnt = time_samples.groupby(by=['date', 'time']).size().rename('count')
    time_samples = time_samples.drop_duplicates(subset=['date', 'time']).merge(cnt, left_on=['date', 'time'],
                                                                               right_index=True).sort_values(
                                                                                                         by=['count'])
    return time_samples.loc[time_samples['count'] > m].reset_index(drop=True).sort_values(by = ['date','time'])

def get_dates(N, wind_data, m):
    """ Utility function for 'validate_windfield()'. Return a list of N tuples (date, time)."""
    filtered_times = filter_times(wind_data, m)
    idx = np.random.choice(len(filtered_times), N, replace=False)
    time_samples = filtered_times.iloc[idx]

    return time_samples

def get_sample(wind_data, sample_date, sample_time):
    """ Utility function for 'validate_windfield()'. Return a DataFrame with the different station measurements taken at
    the given date and time, re-indexed from 0. """
    data_sample = wind_data.loc[(wind_data['time'] == sample_time)  # Sample out a single time-slice in the wind data set
                                & (wind_data['date'] == str(sample_date))]
    data_sample.reset_index(inplace=True)
    data_sample = data_sample.drop(columns=['index'])
    return data_sample

def validate_windfield(wind_field, wind_data: WindDataFrame, number_of_samples = 500, min_stations = 10, get_history = False, verbose = False, seed = None):
    """ Validate the given 'wind_field' model by averaging cross-validation score results over a number of different
    samples (given by the 'number_of_samples' argument) from the given 'wind_data' DataFrame. To filter out time-samples
    that have under a minimum number of stations, which can cause problems in the cross-validation routine, one can use
    the 'min_stations' keyword argument  (default: 'min_stations = 10').

    By default, 'number_of_samples = 500'.

    If keyword 'get_history = False' it returns the following DataFrame:
    df =
           measure   score        stddev
    0         RMSE     val           val
    1          R^2     val           val

    where RMSE is the mean Root Mean Square Error averaged over the 500 time-samples, and R^2 is the proportion of
    explained variance.

    If 'verbose = True' it prints out the current time-sample date and time. Default is 'verbose = False'.

    The keyword 'seed' allows the user to set a random seed for the random number generator. Default is 'seed = None'.

    If 'get_history = True' it returns a larger DataFrame containing the history of the performance measures during
    training. NOTE: 'get_history' is only applicable to the MLPWindfield model, and should not be used for other models.
    """
    np.random.seed(seed)
    n = number_of_samples
    dates_and_times = get_dates(n, wind_data, m = min_stations)     # Randomly select n different data points to be used for sampling
    if verbose is True:
        print('Initiating validation routine on model')
        print('Number of data samples: '+str(n))
    zero_scores = np.zeros(n)
    wind_field_scores = np.zeros(n)
    train_scores_history = [k for k in range(n)]
    test_scores_history = [k for k in range(n)]
    div_history = [k for k in range(n)]
    for i in range(n):                             # Iterate over all samples
        sample_date = dates_and_times['date'].iloc[i]     # yyyy-mm-dd
        sample_time = dates_and_times['time'].iloc[i]     # hh:mm
        if verbose is True:
            print('Current sample: '+str(i+1)+' ('+str(sample_date)+' '+str(sample_time)+')')
        data_sample = get_sample(wind_data, sample_date, sample_time)

        # Get scores calculated on this one sample:
        zero_wff = ZeroWindfield()
        zero_scores[i] = score_windfield(zero_wff, data_sample)
        if get_history is False:
           wind_field_scores[i] = score_windfield(wind_field, data_sample) #standard version

        else:
            wind_field_scores[i], train_scores_history[i], test_scores_history[i], div_history[i] = score_windfield_mlp(wind_field, data_sample) #for mlp to get histories

    # Calculate final scores averaged over all samples:
    if n == 1:
        d = {'measure': ['RMSE', 'R^2'],
             'score': [np.sqrt(wind_field_scores[i]), 1-wind_field_scores[0]/zero_scores[0]],
             'stddev': [np.nan, np.nan]}
        return pd.DataFrame(data=d)
    else:
        wind_field_error = stats.sem(np.sqrt(wind_field_scores))
        wind_field_mean_rmse_scores = np.mean(np.sqrt(wind_field_scores))

        # Calculate normalized measure ("R^2").
        mean_windfield = np.mean(wind_field_scores)
        sem_windfield = stats.sem(wind_field_scores)

        mean_zerofield = np.mean(zero_scores)
        #mean_zerofield = np.mean(wind_data['u'].values**2 + wind_data['v'].values**2) #un-hash to calculate variance over all data
        sem_zerofield = stats.sem(zero_scores)
        #sem_zerofield = stats.sem(wind_data['u'].values**2 + wind_data['v'].values**2) #un-hash to calculate variance over all data

        wind_field_over_z = mean_windfield / mean_zerofield  # 1 - R^2 estimate.
        wind_field_over_z_error = np.sqrt((sem_windfield / mean_zerofield)**2+((mean_windfield*sem_zerofield) / mean_zerofield**2)**2)

        d = {'measure': ['RMSE', 'R^2'],
             'score': [wind_field_mean_rmse_scores, 1-wind_field_over_z],
             'stddev' : [wind_field_error, wind_field_over_z_error]}


    if get_history is False:
        return pd.DataFrame(data=d)
    else:
        # Calculate mean training loss and test loss scores during training:
        train_unc = stats.sem(np.sqrt(np.array(train_scores_history)), axis = 0)
        train_history = np.mean(np.sqrt(np.array(train_scores_history)), axis = 0)
        test_unc = stats.sem(np.sqrt(np.array(test_scores_history)), axis = 0)
        test_history = np.mean(np.sqrt(np.array(test_scores_history)), axis = 0)

        # R^2 statistic:
        sem_windfield = stats.sem(np.array(test_scores_history), axis=0)
        R_sq_unc = np.sqrt((sem_windfield / mean_zerofield)**2 +((np.mean(np.array(test_scores_history), axis=0)*sem_zerofield) / mean_zerofield**2)**2)
        R_sq_history = 1-np.mean(np.array(test_scores_history), axis=0)/mean_zerofield

        # Div statistics:
        div = np.mean(np.array(div_history), axis =0)
        div_unc = stats.sem(np.array(div_history), axis =0)

        history = pd.DataFrame({'train_loss': train_history,
                       'train_loss_std' : train_unc,
                       'test_loss': test_history,
                       'test_loss_std' : test_unc,
                       'R^2' : R_sq_history,
                       'R^2_std' : R_sq_unc,
                       'div' : div,
                       'div_std' : div_unc})

        return pd.DataFrame(data=d), history

def score_windfield_mlp(wind_field: Windfield,
                    wind_data: WindDataFrame,
                    n_splits=5,
                    random_state=100):
    """ Utility function for validate_windfield(). Return the MSE at the last epoch as well as
    the MSE train- and test-loss histories recorded throughout training. """
    train_history_list = []
    test_history_list = []
    wind_data_index = wind_data.index
    kfcv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    error = []
    for train_idx, test_idx in kfcv.split(wind_data_index):
        # Add the portion of data not used for training as test_data:
        wind_field.test_data = wind_data.loc[test_idx]

        # Fit model and return histories:
        train_mse_history, test_mse_history, div_history = wind_field.fit(wind_data.loc[train_idx])

        # Calculate regular ae:
        test_wind_data = wind_data.loc[test_idx]
        ae = average_square_error(test_wind_data, wind_field)
        error.append(ae)

        # Store training histories for each cross-validation:
        train_history_list.append(train_mse_history)
        test_history_list.append(test_mse_history)

    # Compute mean cross-validation errors (mean MSE) for the histories:
    train_error = np.mean(2*np.array(train_history_list), axis=0)
    test_error = np.mean(np.array(test_history_list), axis=0)

    return np.mean(error), train_error, test_error, div_history

def best_results_from_history(history):
    """ Return the epoch corresponding to the best scores recorded throughout training, as well as a pandas dataframe
    holding these best scores. """
    min_epoch = history["test_loss"].idxmin(skipna=True)
    rmse = history["test_loss"][min_epoch]
    Rsq = history["R^2"][min_epoch]
    rmse_std = history["test_loss_std"][min_epoch]
    Rsq_std  = history["R^2_std"][min_epoch]
    return min_epoch, pd.DataFrame({'measure': ["RMSE", "R^2"],
                                    'score': [rmse, Rsq],
                                    'stddev': [rmse_std, Rsq_std]})
