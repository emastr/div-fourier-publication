from sklearn.model_selection import KFold
from framework.windfield import WindDataFrame, Windfield
import numpy as np
import random as rand
import pandas as pd
import time
import datetime as dt
from framework.tools import date_time_to_datetime
from framework.windfield import WindDataFrame
import os
import framework.data_loader as dl
import itertools
from multiprocessing import Pool


class WindPredictionFrame(WindDataFrame):
    """ A DataFrame that contains wind data and predictions for each
    data point. Can have the same columns as a WindDataFrame but also has
    to have two additional columns with predictions:

    u_pred - The predictions for u
    v_pred - The predictions for v
    """

    def __init__(self, *args, **kwargs):
        super(WindPredictionFrame, self).__init__(*args, **kwargs)

    @staticmethod
    def concatenate(frames):
        frame = pd.concat(frames)
        frame.__class__ = WindPredictionFrame
        assert isinstance(frame, WindPredictionFrame)
        return frame


def split_and_save_data(data, data_folder):
    """
    Save measurement data containing several dates as several hourly snapshots in a desired folder.
    :param data:
    :param data_folder:
    :return:
    """
    data = date_time_to_datetime(data)

    def time_to_string(time):
        return str(time).replace(" ", "_").replace(":", "-")

    def save_slice(slice: pd.DataFrame):
        time = slice.datetime.iloc[0]
        slice.reset_index().to_parquet(data_folder + time_to_string(time) + ".parquet")

    # Save all time slices as pandas
    data.groupby("datetime").apply(save_slice)

    # Save a vector of the time objects
    times = [time_to_string(pd.Timestamp(t)) for t in data.datetime.unique()]
    pd.DataFrame({"time": times}).to_parquet(data_folder + "time_slices.parquet")
    return None


def prediction_errors(prediction_frame: WindPredictionFrame, normalize=False):
    error = prediction_frame.groupby('datetime').apply(lambda f: get_mean_square_error(f, normalize=normalize))
    return pd.DataFrame({'datetime': error.index, 'error': error.values})


def get_mean_square_error(prediction_frame: WindPredictionFrame, normalize=False):
    """Get mean square error from WindDataFrame of measurements and predictions"""
    return np.mean(get_square_error(prediction_frame, normalize=normalize))


def get_square_error(prediction_frame: WindPredictionFrame, normalize=False):
    """Convert WindDataFrame of measurements and predictions to a series of errors"""
    u = prediction_frame.u
    v = prediction_frame.v
    u_pred = prediction_frame.u_pred
    v_pred = prediction_frame.v_pred
    error_u = u - u_pred
    error_v = v - v_pred
    sqe = error_u**2+error_v**2
    if normalize:
        # sqe = sqe.divide(u**2 + v**2)
        sqe = sqe/np.mean(u**2 + v**2)
    return sqe


def add_predictions_to_frame(wind_field: Windfield,
                             test_data: WindDataFrame,
                             train_data: WindDataFrame)\
        -> WindPredictionFrame:
    """Fit wind_field to train_data and predict wind velocities of test_data. Add predictions
        as new columns in test_data."""
    wind_field.fit(train_data)
    predictions = test_data.copy()
    x = test_data.x
    y = test_data.y
    pred_wind = wind_field.predict(x, y)
    predictions['u_pred'] = pred_wind.u
    predictions['v_pred'] = pred_wind.v
    predictions.__class__ = WindPredictionFrame
    assert isinstance(predictions, WindPredictionFrame)
    return predictions


def do_kfold_predictions(wind_field: Windfield,
                         wind_data: WindDataFrame,
                         n_splits=5,
                         seed=100)\
        -> WindPredictionFrame:
    """Split wind_data into n_splits equally sized batches and run a k-fold validation scheme
        to retrieve the wind velocity predictions.
        Data:
                wind_field:   wind field to run the validation with
                wind_data:    wind data frame to predict
                n_splits:     Number of batches to divide wind_data into
                random_state: seed for the Kfold validation split
        Returns:
                wind_data with two extra columns "v_pred" and "u_pred"
                containing the predictions.
                """
    wind_data_index = wind_data.index
    kfcv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    block_prediction_array = []
    # Do k-fold validation on the wind_data and return the predictions
    fold = 1
    for train_idx, test_idx in kfcv.split(wind_data_index):
        prediction = add_predictions_to_frame(wind_field,
                                              test_data=wind_data.loc[test_idx], train_data=wind_data.loc[train_idx])

        prediction["Fold_index"] = fold * np.ones(len(prediction.index))
        block_prediction_array.append(prediction)
        fold += 1

    return WindPredictionFrame.concatenate(block_prediction_array)


def do_many_predictions(wind_field: Windfield,
                        wind_data: WindDataFrame,
                        time_samples,
                        n_splits: int = 5,
                        verbose=True) \
        -> WindPredictionFrame:
    """Given a Windfield object and WindData containing measurements from different points in time,
       Sample n_samples of date and time from the WindData and run K-fold crossvalidation on the subset
       of measurements in the Wind data from that specific date and time.

       Parameters
       ----------------
       wind_field:    Windfield data object to use in the crossvalidation
       wind_data:     WindDataFrame with data from multiple points in time
                      has to have more than n_samples time points.
       n_splits:       Value to use for K-fold cross validation
       time_samples:  List of datetime objects for which to predict the velocities
       verbose:       If True, will print out elapsed time and expected remaining time.

       Returns
       ---------------
         WindPredictionFrame: Input data plus a velocity prediction and datetime object for each row.
                                x   y   station_id  altitude  u   v   u_pred   v_pred  datetime
                                -   -      -         -        -   -     -         -       -
                                ...
                                -  -       -         -        -   -     -         -       -
         """
    n_samples = len(time_samples)
    wind_data = date_time_to_datetime(wind_data)
    predictions = []
    n = 0
    t = time.time()
    for datetime in time_samples:
        sub_frame = wind_data[wind_data.datetime == datetime].reset_index().drop(columns=['index'])
        predictions.append(do_kfold_predictions(wind_field, sub_frame, n_splits=n_splits, seed=100))
        # Print out details
        t_elapsed = time.time() - t
        n += 1
        if verbose:
            print('Making predictions at datetime: ', pd.to_datetime(datetime).strftime("%d-%b-%Y (%H:%M:%S)"))
            print('{0:.3} % done, elapsed time: {1:.3}, time remaining: {2}'.format(n / n_samples * 100, t_elapsed,
                str(dt.timedelta(seconds=np.round((n_samples - n) / n * t_elapsed, decimals=0)))))
    return WindPredictionFrame.concatenate(predictions).reset_index().drop("index", axis=1)


def do_random_predictions(wind_field: Windfield,
                          wind_data: WindDataFrame,
                          n_splits: int = 5,
                          n_samples: int = 5,
                          seed=100,
                          verbose = False) \
        -> WindPredictionFrame:
    """Given a Windfield object and WindData containing measurements from different points in time,
       Sample n_samples of date and time from the WindData and run K-fold crossvalidation on the subset
       of measurements in the Wind data from that specific date and time.

       Parameters
       ----------------
       wind_field:  Windfield data object to use in the crossvalidation
       wind_data:   WindDataFrame with data from multiple points in time
                    has to have more than n_samples time points.
       n_splits:     Value of K to use for K-fold cross validation.
       n_samples:   Number of samples to draw from wind_data
       seed:        Seed the number generator that draws samples from wind_data.
       verbose:     If True, will print out elapsed time and expected remaining time.

       Returns
       ---------------
         Dataframe: consisting of the original data plus a velocity prediction and datettime object for each row.
                    x   y   station_id  altitude  u   v   u_pred   v_pred  datetime
                    -   -      -         -        -   -     -         -       -
                    ...
                    -  -       -         -        -   -     -         -       -
         """

    rand.seed(seed)
    wind_data = date_time_to_datetime(wind_data)
    time_samples = rand.sample(list(wind_data.datetime.unique()), k=n_samples)
    return do_many_predictions(wind_field, wind_data, time_samples, n_splits=n_splits, verbose=verbose)


def do_5fold_wrapper(args):
    """
    Wrapper for k_fold predictions with n_splits = 5, which can be used for
    handling multiprocessing.

    Parameters
    ----------
    args:  List or iterable with 4 elements:
            wind field,  the windfield to use when making predictions
            time_slice,  a datetime object corresponding to the specific time slice to use
            save_folder, a directory to save the predictions to. Should end with "/"
            data_folder, a directory to grab the data from. Should end with "/"

    Returns
    -------

    """
    wind_field = args[0][0]
    save_folder = args[0][1]
    time_slice = args[1]
    data_folder = args[2]

    if not os.path.isdir(save_folder):
        print("Creating directory: ", save_folder)
        os.makedirs(save_folder)

    pred_name = save_folder + time_slice + ".parquet"
    print("Starting 5-fold validation on time ", pred_name)

    if os.path.isfile(pred_name):
        print("File already exists: ", pred_name)
    else:
        # Load data
        wind_data: WindDataFrame = dl.load_parquet(data_folder + time_slice + ".parquet")
        # Make predictions
        pred: WindPredictionFrame = do_kfold_predictions(wind_field, wind_data, n_splits=5, seed=100)
        # Save predictions
        pred.to_parquet(pred_name)
        print("Done with ", pred_name)
    return None


def do_parallel_5fold(windfields, data_folder, save_folders, n_samples: int = 500, seed: int = 100, processes: int = 2, random=True):
    rand.seed(seed)
    all_time_slices = list(pd.read_parquet(data_folder + "time_slices.parquet").values.flatten())
    if random:
        time_slices = rand.sample(all_time_slices, k=n_samples)
    else:
        time_slices = all_time_slices[0:n_samples]
    args = list(itertools.product(list(zip(windfields, save_folders)), time_slices, [data_folder]))

    Pool(processes=processes).map(do_5fold_wrapper, args)
    return None


