import os
import math
import datetime
import requests
import zipfile
import io
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
from shapely.geometry import LineString
from framework.windfield import WindDataFrame
from framework.windfield import WindGeoData


def load_parquet(path) -> WindDataFrame:
    df = pd.read_parquet(path)
    df.__class__ = WindDataFrame
    return df


def load_shape_file_from_url(url, tempDir='/tmp/w/') -> WindGeoData:
    """ Load a shape file from the given URL, containing a zip.

    Parameters
    ----------
    url: string
        The url to fetch zip file from.
    tempDir: string, optional
        The full path of the directory to temporarily unzip files to, default is
        '/tmp/w/'.

    Returns
    -------
    GeoDataFrame
        A data frame containg all the information from the shape file.
    """

    response = requests.get(url)
    response.raise_for_status()
    zip = zipfile.ZipFile(io.BytesIO(response.content))
    return load_shape_file_from_zip(zip, url, tempDir)


def load_shape_file_from_local_file(path, tempDir='/tmp/w/') -> WindGeoData:
    """ Load a shape file from a zip file in the local file system.

    Parameters
    ----------
    path: string
        The full path to the local zip file.
    tempDir: string, optional
        The full path of the directory to temporarily unzip files to, default is
        '/tmp/w/'.

    Returns
    -------
    GeoDataFrame
        A data frame containg all the information from the shape file.
    """

    zip = zipfile.ZipFile(path)
    return load_shape_file_from_zip(zip, path, tempDir)


def load_shape_file_from_zip(zip, fileName, tempDir='/tmp/w/') -> WindGeoData:
    zip.extractall(path=tempDir)
    wdf = None
    for file in zip.namelist():
        if (file.endswith('.shp')):
            wdf = gpd.read_file(tempDir + '/' + file)
    for file in zip.namelist():
        os.remove(tempDir + '/' + file)  # Always clean, otherwise future downloads won't work properly
    if (wdf is None):
        raise Exception(f"No .shp file found in zip file {fileName}")
    wdf.__class__ = WindGeoData
    return wdf


def load_wind_data_from_database(db_name='wind', db_user='root', db_pass='',
                                 crs='epsg:3006',
                                 minDate='2018-01-01', maxDate='2018-12-31') -> WindGeoData:
    """ Loads wind data from MySQL database

    Loads wind data from the specified MySQL database, using the specified map
    projection. The database is assumed to have tables of a pre-defined format.
    The wind data will be restricted to the given period in time.

    Parameters
    ----------
    db_name: str, optional
        The name of the database to read from, default is 'wind'.
    db_user: str, optional
        The username to use when connecting to the database, default is 'root'
    db_pass: str, optional
        The password to use when connecting to the database, default is empty.
    crs: str, optional
        The coordinate reference system to use, as a proj4 string definition,
        default is 'epsg:3006' which is the Sweref99TM projection.
    minDate: str, optional
        The first date to include data from, in format YYYY-MM-DD, default is
        '2018-01-01'.
    maxDate: str, optional
        The last date to include data from, in format YYYY-MM-DD, default is
        '2018-12-31'.

    Returns
    -------
    WindGeoData
        Tabulated wind data containing the following columns:
        station_id - The unique ID of the meteorological station
        altitude - The altitude (height above sea level) of the meteorological
                   station at ground level.
        height - The height over the ground where the observation was made.
        date - The date of the observation, format YYYY-MM-DD.
        time - The time of the observation, format HH:MM.
        speed - The wind speed of the observation
        geometry - The wind observations, line segments of two points where the
                   first point is the station of the observation, and the angle
                   of the line represents the wind direction.
    """

    connection = create_engine(f"mysql+pymysql://{db_user}:{db_pass}@localhost/{db_name}")
    df = pd.read_sql(f"SELECT s.id AS station_id, altitude, height, latitude, longitude, date, time, direction, speed FROM Stations s, Wind w WHERE s.id = w.stationId AND date >= '{minDate}' AND date <= '{maxDate}' AND directionQuality = 'G' AND speedQuality = 'G'", connection)

    # Convert date to string
    df.date = df.date.apply(lambda d: d.strftime("%Y-%m-%d"))

    # Convert timedelta to time and then to string
    df.time = df.time.apply(lambda t: (datetime.datetime.min + t).time().strftime("%H:%M"))

    def create_direction_vector(longitude, latitude, direction):
        alpha = 2 * math.pi * (270 - direction) / 360
        latitude_rad = 2 * math.pi * latitude / 360

        u = math.cos(alpha)
        v = math.sin(alpha)
        d = 1852 * 60  # 1° latitude difference corresponds to d meters, hence our resulting vector will be around 1 meter long
        delta_long = u / math.cos(latitude_rad) / d
        delta_lat = v / d

        return LineString([(longitude, latitude), (longitude + delta_long, latitude + delta_lat)])

    # geometry = [create_direction_vector(longitude=t[0], latitude=t[1], direction=t[2]) for t in zip(df['longitude'], df['latitude'], df['direction'])]
    geometry = df.apply(lambda row: create_direction_vector(row.longitude, row.latitude, row.direction), axis=1)

    gdf = gpd.GeoDataFrame(df[['station_id', 'altitude', 'height', 'date', 'time', 'speed']], crs='epsg:4326', geometry=geometry)  # epsg:4326 means WGS84 Plate-carée (long-lat) projection
    gdf = gdf.to_crs(crs)
    gdf.__class__ = WindGeoData
    return gdf
