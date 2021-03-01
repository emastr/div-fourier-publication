import numpy as np
import framework.windfield as wf
import framework.tools as tools


class AveragingWindfield(wf.Windfield):
    """
    Inverse distance weighting (IDW) interpolation.
    Given velocity measurements $\{(u_i, x_i)\}_{i=1}^N$,
    the predicted velocity at a point x is
    \[
        u(x) = \sum_{i=1}^N  u_i \omega_i (x)
    \]
    where
    \[
        \omega_i(x) = \frac{|| x - x_i ||^{-p}}{\sum_{i=1}^N|| x - x_i ||^{-p}}.
    \]
    The parameter p is specified upon initialisation.
    """
    def __init__(self, exponent=2):
        self.exponent = exponent
        self.x = np.empty([1])
        self.y = np.empty([1])
        self.u = np.empty([1])
        self.v = np.empty([1])

    def fit(self, calibration_data: wf.WindDataFrame):
        self.x = calibration_data.x
        self.y = calibration_data.y
        self.u = calibration_data.u
        self.v = calibration_data.v

    def predict(self, x, y) -> wf.WindDataFrame:

        def calculate_u(x, y):
            weight = 1 / ((np.abs(self.x - x)) ** 2 + np.abs((self.y - y)) ** 2)
            total_weight = np.sum(weight)
            return np.sum(self.u * weight) / total_weight

        def calculate_v(x, y):
            weight = 1 / ((self.x - x) ** 2 + (self.y - y) ** 2)
            total_weight = np.sum(weight)
            return np.sum(self.v * weight) / total_weight

        u = np.array([calculate_u(a, b) for (a, b) in zip(x, y)])
        v = np.array([calculate_v(a, b) for (a, b) in zip(x, y)])
        return tools.create_wind_data_frame(x, y, u, v)
