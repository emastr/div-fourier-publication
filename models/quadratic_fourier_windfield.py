import numpy as np
import framework.windfield as wf
import framework.tools as tools
from models.fourier_tools.fourier_series import FourierSeries
from models.fourier_tools.fourier_optimization import do_ridge_div_regression
from models.fourier_tools.fourier_optimization import do_sobolev_div_regression


class FourierWindfield(wf.Windfield):
    """
    The FourierWindfield models wind using a truncated fourier series on
    a "square" frequency domain specified by the user. It uses least squares
    regression with regularization to estimate the fourier coefficients.
    """

    def __init__(self, shape=(3, 3), reg_type: str = 'sobolev', reg_param: float = 0, div_param: float = 0):
        """

        Parameters
        ----------
        shape:   The shape variable specifies which fourier frequencies are used.
                 if shape = (M, N), the fourier series uses the following basis functions:

                 { exp(2pi * i * (m*x/dx + n*y/dy) },    where  -(M+1)/2  < m < (M+1)/2
                                                         and    -(N+1)/2  < n < (N+1)/2

                Hence M and N have to be odd, and if we look at the frequency pairs as a
                2D grid, shape determines a MxN  rectangle and picks all frequency pairs
                within this rectangle. For example, shape = (5, 3) will produce these frequencies:

                        n
                        ^
                      2 |     | | | | | | | | | |
                      1 |     | | |x|x|x|x|x| | |
                      0 |     | | |x|x|x|x|x| | |
                     -1 |     | | |x|x|x|x|x| | |
                     -2 |     | | | | | | | | | |
                        |
                        ---------------------------> m
                                 -2 -1 0 1 2 3

        reg_type: The type of regularizing norm used in the loss function.
                    'ridge' will use the L2 norm, resulting in ridge regresion.
                    'sobolev' will use one of the simplest cases of the sobolev norm:

                         || u || +  || du/dx || + || du/dy || + || ddu/ddy2 || + || ddy/ddx2 ||+ ||ddu/dxdy ||

                    as a penalty. The derivatives are rescaled.
        reg_param: A parameter adjusting the penalty of the regularization. Increasing reg_param
                   will result in a smoother, less detailed function.

        div_param: A parameter adjusting the penalty of the regularization on the divergence of the fourier series.
                    If div_param = float('inf'), the divergence penalty is implemented as a constraint:

                    minimize (Mean square error +  reg_param * regularization ),    subject to div(u)  = 0

                    otherwise, div_param regulates the weight of the L2 norm of the divergence:

                    minimize (Mean Square error  +  reg_param * regularization +  div_param * L2 norm of divergence )
        """
        periods = 4*1e6*np.ones(2)
        self.series = FourierSeries(FourierSeries.gen_weights(shape), periods=periods, im_dim=2)
        self.div_param = div_param
        self.reg_type = reg_type
        self.reg_param = reg_param

    def fit(self, calibration_data: wf.WindDataFrame):
        I = len(calibration_data.index)
        x = np.ndarray(shape=(2, I))
        x[0, :] = calibration_data.x.values
        x[1, :] = calibration_data.y.values

        f = np.ndarray(shape=(2, I))
        f[0, :] = calibration_data.u.values
        f[1, :] = calibration_data.v.values

        if self.reg_type == 'sobolev':
            self.series.set_coefficients(
                do_sobolev_div_regression(self.series, x, f, sobolev_param=self.reg_param, div_param=self.div_param))
        elif self.reg_type == 'ridge':
            self.series.set_coefficients(
                do_ridge_div_regression(self.series, x, f, ridge_param=self.reg_param, div_param=self.div_param))
        else:
            assert False, "Invalid regularization type: {}. Must be \'ridge\' or \'sobolev\'.".format(self.reg_type)

    def predict(self, x, y) -> wf.WindDataFrame:
        r = np.ndarray(shape=(2, len(x.values)))
        r[0, :] = x.values
        r[1, :] = y.values

        f = np.real(self.series.eval(r))
        return tools.create_wind_data_frame(x, y, f[0, :], f[1, :])
