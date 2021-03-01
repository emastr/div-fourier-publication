from pandas import DataFrame
from geopandas import GeoDataFrame


class WindDataFrame(DataFrame):
    """ A DataFrame that contains wind data suitable for calculations.

    This data structure is a simple Pandas DataFrame that contains prepared
    wind data suitable for calculations. The data may come either from real
    observations in a meteorological station, or from predicted wind data from
    some model.

    The data held in this structure has an underlying map projection that can
    not be changed. Please use a WindGeoData object if you want to be able to
    re-project.

    When using a WindDataFrame object, we adhere to the convention that the data
    frame contains the following columns:

    x - The x coordinates of the data points.

    y - The y coordinates of the data points.

    u - Consider the wind at a given point as a vector whose direction shows the
        wind direction with respect to the x-y-coordinate system used, and whose
        length is the speed in the given unit (typically m/s). This column
        contains the x component of those vectors.

    v - Consider the wind at a given point as a vector whose direction shows the
        wind direction with respect to the x-y-coordinate system used, and whose
        length is the speed in the given unit (typically m/s). This column
        contains the y component of those vectors.

    station_id (optional) - The unique ID of the meteorological station where
                            the wind observation was made.

    altitude (optional) - The altitude (height above sea level) of the
                          ground at the point x, y.

    height (optional) - The height over the ground where the given wind is
                        valid.

    date (optional) - The date of the observation, format YYYY-MM-DD.

    time (optional) - The time of the observation, format HH:MM.
    """

    def __init__(self, *args, **kwargs):
        super(WindDataFrame, self).__init__(*args, **kwargs)


class WindGeoData(GeoDataFrame):
    """ An extension of a GeoDataFrame for representing wind observations.

    A WindGeoData structure is simply a GeoDataFrame whose rows correspond to
    individual wind measurements at a certain geographical location. A
    measurement is typically a real observation from a meteorological station.
    Being a GeoDataFrame, we cater for different map projections of the wind
    observations; changing the projection will work correctly on this structure.

    When using a WindGeoData object, we adhere to the convention that the data
    frame contains the following columns:

    station_id (optional) - The unique ID of the meteorological station where
                            the wind observation was made.

    altitude (optional) - The altitude (height above sea level) of the
                          meteorological station at ground level.

    height (optional) - The height over the ground where the observation was
                        made.

    date (optional) - The date of the observation, format YYYY-MM-DD.

    time (optional) - The time of the observation, format HH:MM.

    speed - The wind speed of the observation, in whatever unit is currently
            used, typically m/s.

    geometry - Line segments consiting of two points where the first point is
               the geographical location of the measurement, and where the angle
               of the line represents the wind direction. The line segment
               should be shapely.geometry.LineString objects.

    Note that while it is technically possible to put other geometrical objects
    in the geometry column, we should never put anything else than line segments
    consisting of two points, or the behavior of this framework is be
    unspecified.
    """

    def __init__(self, *args, **kwargs):
        super(WindGeoData, self).__init__(*args, **kwargs)


class Windfield:
    """Base class for estimators/predictors of a windfield.
    """

    def fit(self, calibration_data: WindDataFrame):
        """ Adapt the windfield to the given calibration data.
        """
        pass

    def predict(self, x, y) -> WindDataFrame:
        """ Returns a WindDataFrame that models wind at the given points.

        Parameters
        ----------
        x: pandas.Series
            The x coordinates of the points for which to predict the wind.
        y: pandas.Series
            The y coordinates of the point for which to predict the wind.

        Returns
        -------
        WindDataFrame
            The modelled/predicted wind data field.
        """
        pass
