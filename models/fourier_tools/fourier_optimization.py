import numpy as np
import scipy as sp
import math
from models.fourier_tools.fourier_series import FourierSeries


def do_ridge_div_regression(series: FourierSeries, x: np.ndarray, f: np.ndarray, ridge_param=0, div_param=0):
    """
    Parameters
    ----------
    series: FourierSeries to optimize coefficients of
    x: (d, I) array with data coordinates
    f: (D, I) array with target data values
    ridge_param: regularizing parameter

    Returns
    -------
    Optimal coefficients using ridge-regression
    """
    Q, p = __get_ridge_matrices(series, x, f, ridge_param)
    return __solve_with_div_constraint(series, Q, p, div_param)


def do_sobolev_div_regression(series: FourierSeries, x: np.ndarray, f: np.ndarray, sobolev_param=0, div_param=0):
    """
        Parameters
        ----------
        series: FourierSeries to optimize coefficients of
        x: (d, I) array with data coordinates
        f: (D, I) array with target data values
        ridge_param: regularizing parameter

        Returns
        -------
        Optimal coefficients using ridge-regression
        """
    Q, p = __get_sobolev_matrices(series, x, f, sobolev_param, k=2)
    return __solve_with_div_constraint(series, Q, p, div_param)


def __solve_with_div_constraint(series: FourierSeries, Q: np.ndarray, p: np.ndarray, div_param=0):
    L = series.num_terms
    D = series.D

    Q_ravel, p_ravel = __unravel_matrices(Q, p)  # Unravel since F are coupled by the divergence

    if math.isinf(div_param):
        div = __get_div_free_matrix(series)
        k = len(div)
        A = np.vstack([np.hstack([Q_ravel, div.T]),
                       np.hstack([div, np.zeros(shape=(k, k), dtype=complex)])])
        b = np.zeros(shape=(D * L + k), dtype=complex)
        b[0:D*L] = p_ravel

        coeff = __reravel_matrix(np.linalg.solve(A, b)[0:D * L], L, D)
    else:
        Q_ravel += div_param * __get_div_penalty_matrix(series)
        coeff = __reravel_matrix(np.linalg.solve(Q_ravel, p_ravel), L, D)
    return coeff


def __get_div_penalty_matrix(series):
    """
    Adding a penalty |m'Fm|^2 to a quadratic minimization problem result in the different dimensions becoming coupled.
    Let F_m be the coefficients of the series, m is a multi-index with L possible values. If F_m is unraveled in a
    vector [(F1_m11, F1_m21, ... , F1_mL1), (F2_m12, ... ),... , (FD_m1D, ... FD_mLD)] = F*_m, The linear system that solves
    the minimization can be expressed by adding a matrix div onto the least squares regression matrix. This
    function returns that matrix.
    ----------
    series: FourierSeries to minimize with the divergence penalty

    Returns
    -------
    div: np.ndarray square matrix to add to the least squares/sobolev/ridge regression matrices.
    """
    assert (series.d == series.D), "Series must be of type R^d -> R^d, i.e. d=D."
    L = series.num_terms
    D = series.D
    div = np.zeros(shape=(D*L, D*L), dtype=complex)
    for i in range(L):
        w = series.weights[i]
        for d1 in range(D):
            for d2 in range(D):
                div[d1*L+i, d2*L+i] = w[d1]*w[d2]
    return div


def __get_div_free_matrix(series: FourierSeries):
    """
    Requiring that the series is free of divergence results in the relation
    m^T * F_m  = 0  where F_m are the coefficients of the series and m is a multi-index with L possible values.
    If F_m is unraveled in a vector [(F1_m1, F1_m2, ... , F1_mL), (F2_m1, ... ),... , (FD_m1, ... FD_mL)] = F*_m,
    This relation can be expressed as div*F = 0, where div is a matrix. Of course, if mi = (0,0,0,..,0) for some i,
    then this row will cause the matrix to become singular and is therefore removed.

    Parameters
    ----------
    series: Series to pose divfree condition on

    Returns
    -------
    div: np.ndarray (l, DL) matrix such that div*F = 0
    """
    assert (series.d == series.D), "Series must be of type R^d -> R^d, i.e. d=D."
    D = series.D
    L = series.num_terms
    div = np.zeros(shape=(L, D*L), dtype=complex)
    k = 0
    for i in range(L):
        weight = series.weights[i]
        if not (weight == 0).all():
            for d in range(D):
                div[k, d*L+i] = weight[d]
            k += 1 # Keep track of number of rows. (needed since we skip m=0)
    return div[0:k]


def __get_sobolev_matrices(series: FourierSeries, x: np.ndarray, f: np.ndarray, param=0, k=2):
    """
    Parameters
    ----------
    series: FourierSeries to optimize coefficients of
    x: (d, I) array with data coordinates
    f: (D, I) array with target data values
    ridge_param: regularizing parameter

    Returns
    -------
    matrices Q, p such that the optimal coefficients c are given by the relation Qc = p
    """
    Q, p = __get_least_square_matrices(series, x, f)
    for i in range(len(Q)):
        w = np.abs(series.weights[i, :])
        Q[i, i] += param*__get_sobolev_coeff(w, k=k)
    return Q, p


def __get_sobolev_coeff(weight, k=2):
    # Base case
    if k == 0:
        return 1
    else:
        coeff = __get_sobolev_coeff(weight, k=k-1)
        return np.sum(weight*coeff)+1
    # sum((m,n)*sum((m,n)*1 + 1)) + 1 = sum((m,n)*(m+n+1)) = m*m + 2 m*n + n*n + m + n + 1


def __get_ridge_matrices(series: FourierSeries, x: np.ndarray, f: np.ndarray, ridge_param=0):
    """
    Parameters
    ----------
    series: FourierSeries to optimize coefficients of
    x: (d, I) array with data coordinates
    f: (D, I) array with target data values
    ridge_param: regularizing parameter

    Returns
    -------
    matrices Q, p such that the optimal coefficients c are given by the relation Qc = p
    """
    Q, p = __get_least_square_matrices(series, x, f)
    Q += np.eye(len(Q))*ridge_param
    return Q, p


def __unravel_matrices(Q, p):
    D = len(p[0])
    L = len(Q)
    Q_ravel = np.kron(np.eye(D), Q)
    p_ravel = np.zeros((L * D), dtype=complex)
    for d in range(D):
        p_ravel[d * L:(d + 1) * L] = p[:, d]
    return Q_ravel, p_ravel


def __reravel_matrix(b_ravel, L, D):
    assert len(b_ravel) == D*L, "b_ravel must have length D*L"
    b = np.zeros((L, D), dtype=complex)
    for d in range(D):
        b[:, d] = b_ravel[d*L:(d+1)*L].squeeze()
    return b


def __get_least_square_matrices(series: FourierSeries, x: np.ndarray, f: np.ndarray):
    """
    Parameters
    ----------
    series: FourierSeries to optimize coefficients of
    x: (d, I) array with data coordinates
    f: (D, I) array with target data values
    ridge_param: regularizing parameter

    Returns
    -------
    matrices Q, p such that the optimal coefficients c are given by the relation Qc = p
    """
    I = len(x[0, :])
    D = series.D

    assert len(f) == D, "f must have the same dimension as the image of series"
    assert len(x[0, :]) == len(f[0, :]), "x and f must have equally many entries."

    E = FourierSeries.get_fourier_terms(series.weights, series.periods, x).T    #(I, num_terms)
    EH = E.conj().T

    Q = np.matmul(EH, E)/I
    p = np.matmul(EH, f.T)/I
    return Q, p