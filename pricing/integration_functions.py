import numpy as np


def black_scholes_characteristic_function(u, x, r, sigma, mt):
    y = np.exp(1j * u * (np.log(x) + r * mt) - (1j * u + u ** 2) * sigma ** 2 / 2 * mt)
    return y


def heston_characteristic_function(u, x, gamma, r, sigma_tilde, mt, t, kappa, lamb):
    # Characteristic function of log(spot(mt)) in the Heston model
    def d(u):
        return np.sqrt(lamb ** 2 + sigma_tilde ** 2 * (1j * u + u ** 2))

    y1 = np.exp(1j * u * (x + r * (mt-t)))
    y2 = ((np.exp(lamb * (mt-t) / 2)) / (np.cosh(d(u) * (mt-t) / 2) + lamb * np.sinh(d(u) * (mt-t) / 2) / d(u))) ** (
                2 * kappa / sigma_tilde ** 2)
    y3 = np.exp(-gamma * ((1j * u + u ** 2) * np.sinh(d(u) * (mt-t) / 2) / d(u)) / (
                np.cosh(d(u) * (mt-t) / 2) + lamb * np.sinh(d(u) * (mt-t) / 2) / d(u)))
    return y1 * y2 * y3


def laplace_transform_vanilla(z, strike):
    # Laplace transform of the function f(x) = max(e^x - K, 0) or f(x) = max(K - e^x, 0)
    return (strike ** (1 - z)) / (z * (z - 1))
