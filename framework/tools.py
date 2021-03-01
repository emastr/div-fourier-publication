import math
import pandas as pd
from framework.windfield import WindDataFrame
from framework.windfield import WindGeoData


def create_wind_data_frame(x, y, u, v) -> WindDataFrame:
    wdf = WindDataFrame()
    wdf['x'] = x
    wdf['y'] = y
    wdf['u'] = u
    wdf['v'] = v
    return wdf


def convert_wind_geo_data_to_wind_data_frame(wind_data: WindGeoData) -> WindDataFrame:
    x = get_x(wind_data)
    y = get_y(wind_data)
    u = get_u(wind_data)
    v = get_v(wind_data)
    wdf = create_wind_data_frame(x, y, u, v)

    if 'station_id' in wind_data.columns:
        wdf['station_id'] = wind_data['station_id']
    if 'altitude' in wind_data.columns:
        wdf['altitude'] = wind_data['altitude']
    if 'height' in wind_data.columns:
        wdf['height'] = wind_data['height']
    if 'date' in wind_data.columns:
        wdf['date'] = wind_data['date']
    if 'time' in wind_data.columns:
        wdf['time'] = wind_data['time']

    return wdf


def get_x(wind_data: WindGeoData) -> pd.Series:
    return wind_data['geometry'].apply(lambda vector: vector.coords[0][0])


def get_y(wind_data: WindGeoData) -> pd.Series:
    return wind_data['geometry'].apply(lambda vector: vector.coords[0][1])


def get_u(wind_data: WindGeoData) -> pd.Series:
    def u(row):
        vector = row.geometry
        dx = vector.coords[1][0] - vector.coords[0][0]
        dy = vector.coords[1][1] - vector.coords[0][1]
        length = math.sqrt(dx * dx + dy * dy)
        return row.speed * dx / length if (length > 0) else 0
    return wind_data.apply(u, axis=1)


def get_v(wind_data: WindGeoData) -> pd.Series:
    def v(row):
        vector = row.geometry
        dx = vector.coords[1][0] - vector.coords[0][0]
        dy = vector.coords[1][1] - vector.coords[0][1]
        length = math.sqrt(dx * dx + dy * dy)
        return row.speed * dy / length if (length > 0) else 0
    return wind_data.apply(v, axis=1)


def date_time_to_datetime(frame):
    """
    Add a column of datetime objects to a dataframe with "date" and "time" columns.
    :param frame: pandas.DataFrame containing rows "date" (format MM/DD) and "time" (format HH:mm)
    :return: The same frame with an additional column of datetime objects.
    """
    t = frame['date'] + " " + frame['time'] + ":00"
    frame['datetime'] = pd.to_datetime(t)
    return frame