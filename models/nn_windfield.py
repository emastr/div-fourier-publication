import numpy as np

import framework.windfield as wf
import framework.tools as tools


class NearestNeighbourWindfield(wf.Windfield):
    """
    Nearest Neighbors interpolation of wind velocities.
    Given measurements $\{(u_i, x_i)\}_{i=1}^N$, the
    estimate of u at a point x is u_j, where
    \[
        j = \argmin_i \{ || x - x_i || \}
    \]
    """
    def __init__(self):
        self.x = np.empty([1])
        self.y = np.empty([1])
        self.u = np.empty([1])
        self.v = np.empty([1])
        self.l = -1

    def fit(self, calibration_data: wf.WindDataFrame):
        self.l = len(calibration_data.index)
        self.x = np.array(calibration_data.x).reshape((self.l, 1))
        self.y = np.array(calibration_data.y).reshape((self.l, 1))
        self.u = np.array(calibration_data.u).reshape((self.l,))
        self.v = np.array(calibration_data.v).reshape((self.l,))

    def predict(self, x, y) -> wf.WindDataFrame:
        l2 = len(x)
        x_m = np.array(x).reshape((1, l2))
        y_m = np.array(y).reshape((1, l2))

        d_x = np.repeat(self.x, l2, axis=1) - np.repeat(x_m, self.l, axis=0)
        d_y = np.repeat(self.y, l2, axis=1) - np.repeat(y_m, self.l, axis=0)

        distance = d_x ** 2.0 + d_y ** 2.0

        nearest = np.argmin(distance, axis=0)

        u = self.u[nearest]
        v = self.v[nearest]

        return tools.create_wind_data_frame(x, y, u, v)
