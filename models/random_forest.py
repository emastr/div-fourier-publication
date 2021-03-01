import framework.windfield as wf
import framework.tools as tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd


class RandomForestField(wf.Windfield):
    """
    Random forest (RF) interpolation of a feature map of the coordinates, including altitudes.
    The random forest has proven useful in geospatial interpolation before, see (Appelhans, 2015) and (Hengl, 2018).
    It works by fitting an ensemble of regression trees and taking the average. See the notebook for a guide.
    """
    def __init__(self, altitude_lookup, poly_degree=4, **forest_args):
        """
        Create a random forest field interpolation object
        :param altitude_lookup: Lookup function for the altitudes. Takes numpy arrays x, y and outputs altitude z.
        :param poly_degree: Degree of polynomial features mapping.
        :param forest_args: Arguments used in the RandomForestRegressor from sklearn.
        """
        self.rf = RandomForestRegressor(**forest_args)
        self.pol = PolynomialFeatures(poly_degree)
        self.altitude_lookup = altitude_lookup

    def fit(self, calibration_data: wf.WindDataFrame):
        df = calibration_data.copy()
        df['z'] = self.altitude_lookup(df.x, df.y)
        self.rf.fit(X=self.pol.fit_transform(df[['x', 'y', 'z']]), y=df[['u', 'v']])

    def predict(self, x, y) -> wf.WindDataFrame:
        df = pd.DataFrame({'x': x, 'y': y})
        df['z'] = self.altitude_lookup(df.x, df.y)

        uv = self.rf.predict(X=self.pol.fit_transform(df[['x', 'y', 'z']]))

        frame = tools.create_wind_data_frame(df.x, df.y, uv[:, 0], uv[:, 1])
        frame['altitude'] = df.z
        return frame

    @staticmethod
    def make_random_forest_field(frame, poly_degree=4, **forest_args):
        lu = RandomForestField.make_altitude_lookup(frame)
        return RandomForestField(lu, poly_degree, **forest_args)

    @staticmethod
    def make_altitude_lookup(frame: pd.DataFrame):
        frame = frame[['x', 'y', 'altitude']].drop_duplicates()
        l1 = len(frame.index)

        x_ref = np.array(frame['x']).reshape((l1, 1))
        y_ref = np.array(frame['y']).reshape((l1, 1))

        def lookup(x, y):
            # Copied code from nn_windfield
            l2 = len(x)
            x_m = np.array(x).reshape((1, l2))
            y_m = np.array(y).reshape((1, l2))

            d_x = np.repeat(x_ref, l2, axis=1) - np.repeat(x_m, l1, axis=0)
            d_y = np.repeat(y_ref, l2, axis=1) - np.repeat(y_m, l1, axis=0)

            distance = d_x ** 2.0 + d_y ** 2.0

            nearest = np.argmin(distance, axis=0)
            return frame.iloc[nearest]['altitude'].values

        return lookup



