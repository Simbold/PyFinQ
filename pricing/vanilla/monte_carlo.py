import numpy as np
import numba
from scipy.stats import norm
from monte_carlo import mc_bs_eu_numba, longstaff_schwartz, mc_bs_eu_is_numba


@numba.jit(nopython=True, parallel=False)
def call_payoff_numba(x, strike):
    return np.maximum(x - strike, 0)


@numba.jit(nopython=True, parallel=False)
def put_payoff_numba(x, strike):
    return np.maximum(strike - x, 0)


def monte_carlo_bs_eu(spot, strike, r, d, sigma, mt, n, option_type, antithetic, importance_sampling=False, mu=None, alpha=0.05):

    n2 = int(n / 2)
    n1 = n2 * 2

    if (mu is None) & (importance_sampling is True):
        mu = (np.log(strike/spot) - (r-0.5*sigma**2)*mt)/(sigma*np.sqrt(mt))  # if mu not specified we use d=mu for is.

    if (d != 0) & (importance_sampling is True):
        print("d is set to zero when importance sampling is used")

    if (option_type == "call") & (not importance_sampling):
        [v0, var] = mc_bs_eu_numba(spot, strike, r, d, sigma, mt, n1, call_payoff_numba, antithetic)
        ci = [v0 - norm.isf(alpha / 2) * np.sqrt(var / (n1-1)), v0 + norm.isf(alpha / 2) * np.sqrt(var / (n1-1))]

    elif (option_type == "put") & (not importance_sampling):
        [v0, var] = mc_bs_eu_numba(spot, strike, r, d, sigma, mt, n1, put_payoff_numba, antithetic)
        ci = [v0 - norm.isf(alpha / 2) * np.sqrt(var / (n1-1)), v0 + norm.isf(alpha / 2) * np.sqrt(var / (n1-1))]

    elif (option_type == "put") & importance_sampling:
        [v0, var] = mc_bs_eu_is_numba(spot, strike, r, sigma, mt, mu, n1, put_payoff_numba, antithetic)
        ci = [v0 - norm.isf(alpha / 2) * np.sqrt(var / (n1-1)), v0 + norm.isf(alpha / 2) * np.sqrt(var / (n1-1))]

    elif (option_type == "call") & importance_sampling:
        [v0, var] = mc_bs_eu_is_numba(spot, strike, r, sigma, mt, mu, n1, call_payoff_numba, antithetic)
        ci = [v0 - norm.isf(alpha / 2) * np.sqrt(var / (n1-1)), v0 + norm.isf(alpha / 2) * np.sqrt(var / (n1-1))]

    else:
        print("ERROR: option_type must be 'call' or 'put' and importance_sampling must be True or False!")
        return None
    return [v0, ci]


def polynomial_basis(x, k, strike):
    A = np.ones((x.shape[1], k + 1), dtype=np.float64)
    for i in range(1, k + 1):
        A[:, i] = x ** i
    return A


def laguerre_basis(x, k, strike):
    A = np.ones((x.shape[1], k + 1), dtype=np.float64)
    if k >= 1:
        A[:, 1] = np.exp(-x / 2)
    if k >= 2:
        A[:, 2] = np.exp(-x / 2) * (1 - x)
    if k >= 3:
        A[:, 3] = np.exp(-x / 2) * (x ** 2 + 4 * x + 2) / 2
    if k >= 4:
        A[:, 4] = np.exp(-x / 2) * (-x ** 3 + 9 * x ** 2 - 18 * x + 6) / 6
    if k >= 5:
        A[:, 5] = np.exp(-x / 2) * (x ** 4 - 16 * x ** 3 + 72 * x ** 2 - 96 * x + 24) / 24
    if k >= 6:
        A[:, 6] = np.exp(-x / 2) * (x ** 5 + 25 * x ** 4 - 200 * x ** 3 + 600 * x ** 2 + 120) / 120
    if k >= 7:
        A[:, 7] = np.exp(-x / 2) * (
                x ** 6 - 36 * x ** 5 + 450 * x ** 4 - 2400 * x ** 3 + 5400 * x ** 2 - 4320 * x + 720) / 720
    if (int(k) == k) | k > 7:
        print("ERROR: requested k not possible, k must be integer between 1 and 7")
        return
    return A


def monte_carlo_bs_am(strike, r, mt, option_type, paths, k, basis="laguerre", fit_method="qr"):

    if option_type == "call":
        payoff = call_payoff_numba
    elif option_type == "put":
        payoff = put_payoff_numba
    else:
        print("ERROR: option_type must be 'call' or 'put'!")
        return None

    if basis == "laguerre":
        basis_function = laguerre_basis
        norm_factor = strike
    elif basis == "polynomial":
        basis_function = polynomial_basis
        norm_factor = 1
    else:
        print("ERROR: requested basis function not available! Use 'laguerre' or 'polynomial'")
        return None

    [v0, se] = longstaff_schwartz(strike, mt, r, paths, k, norm_factor, payoff, basis_function, fit_method, itm=True)

    return [v0, se]