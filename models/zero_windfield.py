import numpy as np
from geopandas import GeoDataFrame

import framework.windfield as wf
import framework.tools as tools


class ZeroWindfield(wf.Windfield):

    def __init__(self):
        pass

    def fit(self, calibration_data: wf.WindDataFrame):
        pass

    def predict(self, x, y) -> wf.WindDataFrame:
        return tools.create_wind_data_frame(x, y, np.zeros_like(x), np.zeros_like(y))
