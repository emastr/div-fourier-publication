import framework.windfield as wf
import pandas as pd
import framework.tools as tools
import numpy as np


class WeightingWindfield(wf.Windfield):
    """
    Windfield that returns a weighted average of the input windfields.
    """

    def __init__(self, windfields, weights):
        self.windfields = windfields
        #sum_w = sum(weights)
        #self.weights = [w/sum_w for w in weights]
        self.weights = weights

    def fit(self, calibration_data: wf.WindDataFrame):
        for wf in self.windfields:
            wf.fit(calibration_data)

    def predict(self, x, y) -> wf.WindDataFrame:
        wdf = tools.create_wind_data_frame(x, y, np.zeros(len(x)), np.zeros(len(x)))
        wdf[['u', 'v']] = sum([w*wf.predict(x, y)[['u','v']] for w, wf in zip(self.weights, self.windfields)])
        return wdf

    @ staticmethod
    def get_weights(windfield_predictions):
        """

        Parameters
        ----------
        windfield_predictions: List of data frames, consisting of predictions made from the different wind fields.
                                the predictions can be generated using do_random_predictions(). Make sure to use
                                the same seed for all models.

        Returns
        -------

        """

        prediction = windfield_predictions[0]

        N = len(prediction.index)

        u = prediction.u.values.reshape((N, 1))
        v = prediction.v.values.reshape((N, 1))

        u_p = []
        v_p = []

        for prediction in windfield_predictions:
            u_p.append(prediction.u_pred.values)
            v_p.append(prediction.v_pred.values)

        return WeightingWindfield.do_linear_regression(u, v, np.array(u_p).T, np.array(v_p).T)

    @ staticmethod
    def do_linear_regression(u, v, u_p, v_p) -> np.ndarray:
        """

        Parameters
        ----------
        u: (I, 1) np.ndarray of velocity data
        v: (I, 1) np.ndarray of velocity data
        u_p: (I x K) np array of model approximations
        v_p: (I x K) np array of model approximations

        Returns
        -------
        weigths: (K x 1) vector minimizing the square error of (y-a'*x)
        """
        return np.linalg.solve(u_p.T @ u_p + v_p.T @ v_p, u_p.T @ u + v_p.T @ v).flatten()