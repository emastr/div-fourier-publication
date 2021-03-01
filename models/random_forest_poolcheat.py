import framework.windfield as wf
import framework.tools as tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd


class RandomForestField(wf.Windfield):
    #def __init__(self, altitude_lookup, n_trees, poly_degree=4):
    def __init__(self, n_trees, lookup_frame, poly_degree=4):
        self.rf = RandomForestRegressor(n_estimators=n_trees)
        self.pol = PolynomialFeatures(poly_degree)
        self.lookup_data = lookup_frame
        #self.altitude_lookup = make_altitude_lookup(lookup_frame)


    def fit(self, calibration_data: wf.WindDataFrame):
        ############
        frame = self.lookup_data
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

        self.altitude_lookup = lookup
        #############
        df = calibration_data.copy()
        df['z'] = self.altitude_lookup(df.x, df.y)
        self.rf.fit(X=self.pol.fit_transform(df[['x', 'y', 'z']]), y=df[['u', 'v']])

    def predict(self, x, y) -> wf.WindDataFrame:
    ############
        frame = self.lookup_data
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
        self.altitude_lookup = lookup
        #############

        df = pd.DataFrame({'x': x, 'y': y})
        df['z'] = self.altitude_lookup(df.x, df.y)

        uv = self.rf.predict(X=self.pol.fit_transform(df[['x', 'y', 'z']]))

        frame = tools.create_wind_data_frame(df.x, df.y, uv[:, 0], uv[:, 1])
        frame['altitude'] = df.z
        return frame

@staticmethod
def make_random_forest_field(frame, n_trees, poly_degree=4):
    lu = make_altitude_lookup(frame)
    return RandomForestField(lu, n_trees, poly_degree)

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



