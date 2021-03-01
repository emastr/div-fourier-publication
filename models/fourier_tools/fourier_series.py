import numpy as np
from typing import List, Tuple


class FourierSeries:
    """Multivariate truncated fourier series f: R^d -> C^D."""

    def __init__(self, weights: np.ndarray, periods: np.array, coefficients: np.ndarray = None, im_dim: int = 1):
        """
        Parameters
        ----------
        weights:      np.ndarray, size=(num_terms, d), dtype=int, an array representing fourier terms
                      each row of weights represents the weights of one fourier term
        periods:      np.array, size=(d,) array of the periods with which to rescale x
        coefficients: np.ndarray, size=(num_terms, D) the coefficients
        im_dim:       If coefficients were not specified, determines the dimension of the output of the series.
        """
        # If d=1, need to make an extra dimension for matrix multiplications to work
        #weights = fix_singleton_dim(weights)
        #coefficients = fix_singleton_dim(coefficients)

        d = len(weights[0])
        assert all([d == len(w) for w in weights]), "All weight vectors must have equal length"
        assert d == len(periods), "The periods and weights must have equal numbers of elements"
        self.d = d                                      # Number of dimensions in position data
        self.num_terms = len(weights)                   # Number of fourier terms
        self.weights = weights.astype(int)              # Weights for all of the terms
        self.set_dict()                                 # Make lookup table for the weights
        self.periods = np.expand_dims(periods, axis=1)  # Period (rescaling) of each dimension, make into (d,1) matrix
        # Coefficients for each term
        if coefficients is not None:
            im_dim = len(coefficients[0])
            assert len(coefficients) == len(weights), "Not equally many weights and coefficients."
            assert all([im_dim == len(c) for c in coefficients]), "the coefficients must all be of equal length"
            self.coeff = coefficients
            self.D = im_dim
        else:
            self.D = im_dim
            self.coeff = np.zeros((self.num_terms, im_dim))

    def contains_weight(self, weight):
        return self.weight_dict.__contains__(weight.astype(int).tobytes())

    def get_weight_index(self, weight):
        assert self.contains_weight(weight), "Weight is not in this series"
        return self.weight_dict[weight.astype(int).tobytes()]

    def get_coeff(self, weight):
        if self.contains_weight(weight):
            return self.coeff[self.get_weight_index(weight)]
        else:
            return np.zeros(shape=(self.D))

    def set_dict(self):
        self.weight_dict = dict()
        for i in range(self.num_terms):
            key = self.weights[i].tobytes()
            self.weight_dict[key] = i

    def lookup_coeff(self, weight):
        return self.weight_dict[weight.tobytes()]

    def set_weights(self, weights):
        assert len(weights[0]) == self.d, "Weights must have dimenstion {}".format(self.d)
        self.weights = weights
        self.num_terms = weights.shape[0]
        self.set_dict()
        self.set_coefficients(np.zeros((self.num_terms, self.D)))

    def set_coefficients(self, coefficients: np.array):
        assert (len(coefficients) == self.num_terms) & (len(coefficients[0]) == self.D), \
            "Coefficients must have shape ({0}, {1})".format(self.num_terms, self.D)
        self.coeff = coefficients

    def get_divergence_series(self):
        """Create new fourier series div(f)."""
        coeff = self.get_divergence_coeff()
        return FourierSeries(self.weights, self.periods.flatten(), coeff)

    def get_rot_series(self):
        """Create new fourier series div(f)."""
        coeff = self.get_rot_coeff()
        return FourierSeries(self.weights, self.periods.flatten(), coeff)

    def get_divergence_coeff(self):
        """Get coefficients of div(f)"""
        assert self.d == self.D, "Divergence is only defined if d=D."
        coeff = np.zeros((self.num_terms, 1), dtype=complex)
        coeff[:, 0] = np.sum(self.weights*self.coeff/self.periods.T*2.0j*np.pi, axis=1)
        return coeff

    def get_rot_coeff(self):
        """Get coefficients of div(f)"""
        assert self.d == 2, "series has to be 2d."
        assert self.d == self.D, "Divergence is only defined if d=D."
        coeff = np.zeros((self.num_terms, 1), dtype=complex)
        rotvec = self.weights @ np.array([[0, -1], [1, 0]])
        coeff[:, 0] = np.sum(rotvec*self.coeff/self.periods.T*2.0j*np.pi, axis=1)
        return coeff

    def eval(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate fourier series in all points specified by x.

        Parameters
        ----------
        x: (d, I) array with points

        Returns
        -------
        val: (D, I) array with evaluations
        """
        assert len(x) == self.d, "x must be {}-dimensional.".format(self.d)
        basis_funcs = FourierSeries.get_fourier_terms(self.weights, self.periods, x)
        # return sum_n e^(2pi weights(n) x/T*i)*coeff(n) -> (D, I) array

        val = np.matmul(self.coeff.T, basis_funcs)
        return val

    def print_series(self):
        print("FourierSeries f: R^{0} -> C^{1} with {2} terms".format(self.d, self.D, self.num_terms))
        print("Weights:")
        print(self.weights.T)
        print("Coefficients:")
        print(self.coeff.T)
        print("Periods:")
        print(self.periods)

    def gen_coeff_matrices(self):
        max_weights = (np.abs(self.weights)).max(axis=0)
        coeff_mats = []
        for d in range(self.D):
            mat = np.zeros(shape=tuple(2*max_weights+1), dtype=complex)
            it = np.nditer(mat, flags=['multi_index'])
            while not it.finished:
                weight = it.multi_index - max_weights
                mat[it.multi_index] = self.get_coeff(weight)[d]
                it.iternext()
            coeff_mats.append(mat)
        return coeff_mats

    @ staticmethod
    def get_fourier_terms(weights, periods, x):
        """
        Parameters
        ----------
        weights: (num_terms, d) ndarray with weights
        periods: (d,1) periods for each coordinate
        x: (d, I) ndarray with I points to calculate

        Returns
        ----------
        (num_terms, I) array with evaluations
        """
        scaled_x = x/periods
        return np.exp(2.0j*np.pi*np.matmul(weights, scaled_x))

    @ staticmethod
    def gen_weights(shape: Tuple[int]) -> np.ndarray:
        shape_ar = np.asarray(shape)
        assert shape_ar.__mod__(2).all(), "The number of weights must be uneven"
        weights = np.zeros((shape_ar.prod(), len(shape)))
        weight_shift = -(shape_ar-1)/2
        it = np.nditer(np.zeros(shape), flags=['multi_index'])
        while not it.finished:
            weights[it.iterindex, :] = (it.multi_index + weight_shift).astype(int)
            it.iternext()
        return weights









