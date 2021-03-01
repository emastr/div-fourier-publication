from framework import tools
from framework import data_loader
from sklearn.model_selection import KFold
from framework.windfield import Windfield, WindDataFrame
from geopandas import GeoDataFrame
import numpy as np
import framework.data_loader as dl


def average_square_error(test_wind_data: WindDataFrame, wind_field: Windfield):
    x = test_wind_data.x  # tools.get_x(test_wind_data)
    y = test_wind_data.y  # tools.get_y(test_wind_data)
    u = test_wind_data.u  # tools.get_u(test_wind_data)
    v = test_wind_data.v  # tools.get_v(test_wind_data)
    est_wind = wind_field.predict(x, y)
    u_est = est_wind.u  # tools.get_u(est_wind).values
    v_est = est_wind.v  # tools.get_v(est_wind).values
    error_u = u - u_est
    error_v = v - v_est
    ase = np.mean(error_u ** 2 + error_v ** 2)
    return ase


def average_R_2_error(test_wind_data: WindDataFrame, wind_field: Windfield):
    x = test_wind_data.x  # tools.get_x(test_wind_data)
    y = test_wind_data.y  # tools.get_y(test_wind_data)
    u = test_wind_data.u  # tools.get_u(test_wind_data)
    v = test_wind_data.v  # tools.get_v(test_wind_data)
    est_wind = wind_field.predict(x, y)
    u_est = est_wind.u  # tools.get_u(est_wind).values
    v_est = est_wind.v  # tools.get_v(est_wind).values
    error_u = u - u_est
    error_v = v - v_est
    r2 = np.mean(error_u ** 2 + error_v ** 2)/np.mean(u**2 + v**2)
    return r2


def score_windfield(wind_field: Windfield,
                    wind_data: WindDataFrame,
                    n_splits=5,
                    random_state=100):
    wind_data_index = wind_data.index
    kfcv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    error = []
    for train_idx, test_idx in kfcv.split(wind_data_index):
        wind_field.fit(wind_data.loc[train_idx])
        test_wind_data = wind_data.loc[test_idx]
        ae = average_square_error(test_wind_data, wind_field)
        error.append(ae)
    return np.mean(error)


