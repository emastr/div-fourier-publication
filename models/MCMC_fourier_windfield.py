import numpy as np
import framework.windfield as wf
from models.quadratic_fourier_windfield import FourierWindfield as FWF


# If this produces bugs, rename to FourierWindfield
class RandomFourierFeatures(wf.Windfield):
    """
    The FourierWindfield models wind using a truncated fourier series on
    a "square" frequency domain specified by the user. It uses least squares
    regression with regularization to estimate the fourier coefficients.
    """

    def __init__(self, n_terms=440, n_steps=100, reg_param: float = 0, div_param: float = 0, gamma=2, sigma=2, seed=100):
        """
        Parameters
        ----------
        :param n_steps:  Number of steps for the Metropolis-Hastings algorithm (Hyper parameter)
        :param n_terms:  Number of Fourier terms
        :param reg_param: weight adjusting the amount of Regularisation (sobolev norm)
        :param div_param: weight adjusting the penalty on high divergence
        :param gamma:   Hyper parameter for the Metropolis algorithm
        :param sigma:   Hyper parameter for the Metropolis algorithm
        :param seed:    Seed for the Metropolis algorithm
        """

        self.seed = seed
        self.gamma = gamma
        self.sigma = sigma
        self.n_steps = n_steps
        self.n_terms = n_terms
        # Initialize windfield
        self.wf = FWF(shape=(1, 1), reg_type='sobolev', reg_param=reg_param, div_param=div_param)

    def fit(self, calibration_data: wf.WindDataFrame):
        wf = self.wf
        gamma = self.gamma
        sigma = self.sigma
        n_terms = self.n_terms
        n_steps = self.n_steps
        np.random.seed(self.seed)
        sqrt_n_terms = n_terms**0.5
        weights = np.random.randint(low=-sqrt_n_terms, high=sqrt_n_terms, size=(n_terms, 2))
        step_dir = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]).astype(int)

        weight_history = []
        coeff_history = []

        # Do first step:
        wf.series.set_weights(np.array(weights))
        wf.fit(calibration_data)
        coef = [wf.series.get_coeff(w) for w in weights]

        def get_steps(sigma, n_terms):
            # Alternative 1: random walk 4 directional
            # idx = np.random.choice([i for i in range(4)], n_terms)  # Index for the step directions for each weight
            # steps = step_dir[idx, :]

            # Alternative 2: random walk normal distribution, round to nearest integer.
            steps = (sigma * np.random.standard_normal(size=(n_terms, 2))).astype(int)
            return steps

        for n in range(n_steps):

            # Create proposal weights
            step = get_steps(sigma, n_terms)
            proposal_weights = weights + step                               # Proposed weights
            weights_to_fit = np.unique(proposal_weights, axis=0)            # Proposed weights, doubles filtered out

            # Fit series with adjacent weights
            wf.series.set_weights(weights_to_fit)                           #
            wf.fit(calibration_data)
            prop_coef = [wf.series.get_coeff(w) for w in proposal_weights]

            # Append to to history
            coeff_history.append(np.array(coef))
            weight_history.append(weights)

            # Do random walk steps
            new_weights = []
            new_coef = []

            for w, w_p, c, c_p in zip(weights, proposal_weights, coef, prop_coef):
                abs_c = np.linalg.norm(c)
                abs_c_p = np.linalg.norm(c_p)
                prob = min(1, (abs_c_p / abs_c)**gamma)

                if np.random.random() < prob:
                    new_weights.append(w_p)
                    new_coef.append(c_p)
                else:
                    new_weights.append(w)
                    new_coef.append(c)

            weights = np.array(new_weights)
            coef = new_coef

        # Do something with history
        # Or do nothing, in which case self.wf will have the last sampled frequencies
        wf.series.set_weights(weights)  #
        wf.fit(calibration_data)

        # Return history
        return np.array(coeff_history), np.array(weight_history)

    def predict(self, x, y) -> wf.WindDataFrame:
        return self.wf.predict(x, y)

    def to_string(self):
        gamma = self.gamma
        sigma = self.sigma
        n_steps = self.n_steps
        n_terms = self.n_terms
        reg_param = self.wf.reg_param
        div_param = self.wf.div_param
        string = "MCMCWF_" + \
                 "terms=" + str(n_terms) +  \
                 "_steps=" + str(n_steps) + \
                 "_sigma=" + "{0}".format(sigma).replace(".", "p") + \
                 "_gamma=" + "{0}".format(gamma).replace(".", "p") + \
                 "_reg=" + "{0}".format(reg_param).replace(".", "p") + \
                 "_div=" + "{0}".format(div_param).replace(".", "p")

        return string