from framework import windfield as wf

class ResidualWindfield(wf.Windfield):

    def __init__(self, windfields):
        """
        Creates a residual windfield. The first windfield
        is fitted to the data, the second windfield is fitted to the
        residuals, the third is fitted to the residuals of the residuals and so on.
        This model is only suited for windfields that are not exact, i.e that
        have nonzero error in the fitted points.

        Parameters
        ----------
        windfields
        """

        if len(windfields) != 2:
            self.base_wf = ResidualWindfield(windfields[:-1])
        else:
            self.base_wf = windfields[0]

        self.res_wf = windfields[-1]


    def fit(self, calibration_data: wf.WindDataFrame):
        x = calibration_data.x
        y = calibration_data.y
        u = calibration_data.u
        v = calibration_data.v

        self.base_wf.fit(calibration_data)
        pred: wf.WindDataFrame = self.base_wf.predict(x, y)

        u_res = u - pred.u
        v_res = v - pred.v

        res_calibration_data = calibration_data.copy()
        res_calibration_data.u = u_res
        res_calibration_data.v = v_res

        self.res_wf.fit(res_calibration_data)

    def predict(self, x, y) -> wf.WindDataFrame:
        pred = self.base_wf.predict(x, y)
        res_pred = self.res_wf.predict(x, y)

        pred.u += res_pred.u
        pred.v += res_pred.v

        return pred

    def get_residual_windfield(self):
        """
        Windfield which predicts the residual of the smaller ResidualWindfield formed by all but the
        last input windfields. his is a great tool for predicting and interpolating the prediction error.

        for example: if self = ResidualWindfield([AveragingWindfield(), RandomForestWindfield()]), then
        self.get_residual_windfield() will return a RandomForestWindfield() which predicts u - u_af  and v-v_af,
        u_af and v_af being the predictions generated from the AveragingWindfield().

        Returns
        -------
        res_wf: windfield
        """
        return self.res_wf