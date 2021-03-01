from pykrige.uk import UniversalKriging
import framework.windfield as wf
from framework.tools import create_wind_data_frame
import numpy as np


class KrigingWindfield(wf.Windfield):
    """
    Universal Kriging (UK) windfield.
    Applied on the velocity components separately.
    The default variogram is spherical, but can be user-specified.
    See the pykrige documentation for more details.
    """
    def __init__(self, **ukrige_args):
        """
        :param **ukrige_args: Parameters for UniversalKriging applied to each of the velocity components
        """
        self.u_kriging = None
        self.v_kriging = None
        self.ukrige_args = ukrige_args

    def fit(self, calibration_data: wf.WindDataFrame):
        x = calibration_data.x.values
        y = calibration_data.y.values
        u = calibration_data.u.values
        v = calibration_data.v.values
        def get_kriger(vel)->UniversalKriging:
            return UniversalKriging(x, y, vel,
                                    verbose=False,
                                    enable_plotting=False,
                                    **self.ukrige_args
                                    )
        self.u_kriging: UniversalKriging = get_kriger(u)
        self.v_kriging: UniversalKriging = get_kriger(v)


    def predict(self, x, y) -> wf.WindDataFrame:
        u, _ = self.u_kriging.execute(style='points', xpoints=x, ypoints=y)
        v, _ = self.v_kriging.execute(style='points', xpoints=x, ypoints=y)
        pred = create_wind_data_frame(x, y, u, v)
        return pred