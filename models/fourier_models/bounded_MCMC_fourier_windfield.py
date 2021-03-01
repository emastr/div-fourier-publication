import numpy as np
import framework.windfield as wf
from models.quadratic_fourier_windfield import FourierWindfield as FWF


class FourierWindfield(wf.Windfield):
    """
    The FourierWindfield models wind using a truncated fourier series on
    a "square" frequency domain specified by the user. It uses least squares
    regression with regularization to estimate the fourier coefficients.
    """

    def __init__(self,
                 n_terms: int = 440,
                 n_steps: int = 1000,
                 verbose: bool = False,
                 n_bound: int = 100,
                 gamma: float = 1.5,
                 sigma: float = 2.5,
                 reg_param: float = 0.017,
                 div_param: float = 0.001,
                 seed: int = 100):
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
        self.n_terms = n_terms
        self.n_steps = n_steps
        self.n_bound = n_bound
        self.gamma = gamma
        self.sigma = sigma
        self.reg_param = reg_param
        self.div_param = div_param
        self.seed = seed
        self.wf = FWF(shape=(1, 1), reg_type='sobolev', reg_param=reg_param, div_param=div_param)
        self.verbose = verbose

    def fit(self, calibration_data: wf.WindDataFrame):

        if self.verbose:
            print("Defining variables...")
        # Define variables
        wf = self.wf
        n_terms = self.n_terms
        n_steps = self.n_steps
        n_bound = self.n_bound
        gamma = self.gamma
        sigma = self.sigma
        np.random.seed(self.seed)

        # Initialize weights
        weights = np.random.randint(low=-n_bound, high=n_bound, size=(n_terms, 2))
        #weights = np.random.normal(loc = 0, scale=n_bound/2, size=(n_terms, 2)).astype(int)

        # Initialize history
        weight_history = []
        coeff_history = []
        dist_history = []
        weight_dist = {}

        # Define proposal density
        def get_steps(sigma, n_terms):
            return (sigma * np.random.standard_normal(size=(n_terms, 2))).astype(int)

        # Define weight distribution update
        def update_dist(weights):
            for w in weights:
                key = w.tobytes()
                if key not in weight_dist:
                    weight_dist[key] = [1, w]
                else:
                    weight_dist[key][0] += 1
            return None

        #
        def resample():
            dist = list(weight_dist.values())
            weights = [d[1] for d in dist]
            prob = np.array([d[0] for d in dist])
            key = np.random.choice(a=range(len(weights)), p=prob/sum(prob))
            return weights[key]

        def add_dist_to_history():
            dist_copy = {}
            for key in weight_dist.keys():
                dist_copy[key] = weight_dist[key].copy()
            dist_history.append(dist_copy)

        # Do first step:
        wf.series.set_weights(np.array(weights))
        wf.fit(calibration_data)
        coef = [wf.series.get_coeff(w) for w in weights]
        update_dist(weights)

        # Do the steps
        if self.verbose:
            print("Running markov chain...")
        for n in range(n_steps):
            if self.verbose and (n % 10)==0:
                print("{0} out of {1} steps done".format(n, n_steps), end="\r")
            # Create proposal weights
            step = get_steps(sigma, n_terms)
            proposal_weights = weights + step                               # Proposed weights
            weights_to_fit = np.unique(proposal_weights, axis=0)            # Proposed weights, doubles filtered out

            # Fit series to proposed weights
            wf.series.set_weights(weights_to_fit)
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
                    if (abs(w_p) > n_bound).any():
                        w_p = resample()
                    new_weights.append(w_p)
                    new_coef.append(c_p)
                else:
                    new_weights.append(w)
                    new_coef.append(c)

            weights = np.array(new_weights)
            update_dist(weights)
            add_dist_to_history()
            coef = new_coef

        # Do something with history
        # Or do nothing, in which case self.wf will have the last sampled frequencies
        wf.series.set_weights(weights)  # Isnt this a problem? np.unique first meybe
        wf.fit(calibration_data)

        # Return history
        return np.array(coeff_history), np.array(weight_history), dist_history

    def predict(self, x, y) -> wf.WindDataFrame:
        return self.wf.predict(x, y)

    def to_string(self):
        gamma = self.gamma
        sigma = self.sigma
        n_steps = self.n_steps
        n_terms = self.n_terms
        reg_param = self.wf.reg_param
        div_param = self.wf.div_param
        string = "MCMCWF/" + \
                 "terms=" + str(n_terms) +  \
                 "/steps=" + str(n_steps) + \
                 "/sigma=" + "{0}".format(sigma).replace(".", "p") + \
                 "/gamma=" + "{0}".format(gamma).replace(".", "p") + \
                 "/reg=" + "{0}".format(reg_param).replace(".", "p") + \
                 "/div=" + "{0}".format(div_param).replace(".", "p") + "/"

        return string