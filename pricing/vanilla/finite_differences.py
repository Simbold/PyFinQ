import numpy as np
import numba
from finite_differences_methods import tridiagonal_invers, brennon_schwartz_algorithm


def fidi_bs_eu(strike, r, sigma, mt, a=-0.5, b=0.5, m=500, nu_max=20000, scheme="cn"):
    # strike: float or integer the strike price
    # r: float/integer the risk free interest rate
    # sigma: float/integer, volatility of the underlying
    # mt: float/integer, time until maturity in years
    # a/b: float/integer, bounds in space of the finite difference scheme
    # m: integer, grid in space
    # nu_max: integer, grid in time
    # scheme:  string corresponding to the fidi scheme: 'explicit', 'implicit' or 'cn' (cn is crank nicolson scheme)

    # returns list of v0 and spot
    # v0: option price values
    # spot: to v0 corresponding spot prices

    # check input

    if (round(nu_max) == nu_max) & (round(m) == m):
        pass
    else:
        print("ERROR: nu_max and m must be interpretable as integer")
        return

    if (scheme == "explicit") | (scheme == "implicit") | (scheme == "cn"):
        # if input correct compute
        [v0, spot] = fidi_bs_eu_numba(strike, r, sigma, mt, a, b, m, nu_max, scheme)
    else:
        print("ERROR: scheme must be either 'explicit', 'implicit' or 'cn'")
        return
    return [v0, spot]


def fidi_bs_american(strike, r, sigma, mt, a=-0.5, b=0.5, m=500, nu_max=20000):
    # strike: float or integer the strike price
    # r: float/integer the risk free interest rate
    # sigma: float/integer, volatility of the underlying
    # mt: float/integer, time until maturity in years
    # a/b: float/integer, bounds in space of the finite difference scheme
    # m: integer, grid in space
    # nu_max: integer, grid in time

    # returns list of v0 and spot
    # v0: option price values
    # spot: to v0 corresponding spot prices
    # ------------------------------------------------------------------------------------------------------------------

    # check input
    if (round(nu_max) == nu_max) & (round(m) == m):
        # if input correct compute
        [v0, spot] = fidi_bs_american_numba(strike, r, sigma, mt, a, b, m, nu_max)
    else:
        print("ERROR: nu_max and m must be interpretable as integer")
        return



    return [v0, spot]


@numba.jit(nopython=True)
def fidi_bs_eu_numba(strike, r, sigma, mt, a, b, m, nu_max, scheme):
    # strike: float or integer the strike price
    # r: float/integer the risk free interest rate
    # sigma: float/integer, volatility of the underlying
    # mt: float/integer, time until maturity in years
    # a/b: float/integer, bounds in space of the finite difference scheme
    # m: integer, grid in space
    # nu_max: integer, grid in time
    # scheme:  string corresponding to the fidi scheme: 'explicit', 'implicit' or 'cn' (cn is crank nicolson scheme)

    # returns list of v0 and spot
    # v0: option price values
    # spot: to v0 corresponding spot prices

    q = 2 * r / sigma ** 2
    dx = ((b - a) / m)
    dt = (sigma**2 * 0.5*mt / nu_max)
    lamb = dt / dx ** 2

    # if explicit scheme chosen, check for stability
    if scheme == "explicit":
        if lamb >= 0.5:
            print("ERROR, Finite difference scheme unstable: lambda >=0.5, try again with larger nu_max")
            return

    # initialize for a call option
    w = np.zeros(m + 1, dtype=np.float64)
    for i in numba.prange(0, m + 1):
        x = a + i * dx
        w[i] = np.maximum(np.exp(0.5 * x * (q + 1)) - np.exp(0.5 * x * (q - 1)), 0)
    w[0] = 0

    if scheme == "explicit":
        for i in numba.prange(1, nu_max+1):
            t = i * dt
            # explicit fidi scheme
            w[1:-1] = lamb * w[0:m - 1] + (1 - 2 * lamb) * w[1:m] + lamb * w[2:m + 1]
            # boundary condition of the call
            w[-1] = np.exp(0.5 * (q + 1) * b + (0.5 * (q + 1)) ** 2 * t) - np.exp(
                0.5 * (q - 1) * b + (0.5 * (q - 1)) ** 2 * t)

    elif scheme == "implicit":
        alpha = np.ones(m - 1, dtype=np.float64) * (1 + 2 * lamb)
        beta = np.ones(m - 2, dtype=np.float64) * (-lamb)
        n = m-1

        for i in numba.prange(1, nu_max + 1):
            t = i * dt
            # boundary condition
            w[-1] = np.exp(0.5 * (q + 1) * b + (0.5 * (q + 1))**2 * t) - np.exp(0.5*(q-1)*b + (0.5 * (q - 1))**2 * t)
            w[-2] = w[-2] + lamb * w[-1]
            # implicit scheme (inversion)
            winv = (w[1:-1])
            alpha_hat = np.zeros(n, dtype=np.float64)
            b_hat = np.zeros(n, dtype=np.float64)

            alpha_hat[0] = alpha[0]
            b_hat[0] = winv[0]

            for u in numba.prange(1, n):
                alpha_hat[u] = alpha[u] - beta[u - 1] * beta[u - 1] / alpha_hat[u - 1]
                b_hat[u] = winv[u] - beta[u - 1] * b_hat[u - 1] / alpha_hat[u - 1]

            xinv = np.ones(n, dtype=np.float64)
            xinv[-1] = b_hat[-1] / alpha_hat[-1]

            for u in numba.prange(n - 2, -1, -1):
                xinv[u] = (b_hat[u] - beta[u] * xinv[u + 1]) / alpha_hat[u]
            w[1:-1] = xinv

    elif scheme == "cn":
        alpha = np.ones(m - 1, dtype=np.float64) * (1 + lamb)
        beta = np.ones(m - 2, dtype=np.float64) * (-0.5 * lamb)
        n = len(alpha)

        for i in range(1, nu_max + 1):
            t = i * dt
            # explicit part
            w[1:-1] = 0.5 * lamb * w[0:m - 1] + (1 - lamb) * w[1:m] + 0.5 * lamb * w[2:m + 1]
            # adjusting for boundary condition
            w[-1] = np.exp(0.5 * (q + 1) * b + (0.5 * (q + 1)) ** 2 * t) - np.exp(
                0.5 * (q - 1) * b + (0.5 * (q - 1)) ** 2 * t)
            w[-2] = w[-2] + 0.5 * lamb * w[-1]
            # implicit part (tridiagonal inversion)
            winv = (w[1:-1])
            alpha_hat = np.zeros(n, dtype=np.float64)
            b_hat = np.zeros(n, dtype=np.float64)

            alpha_hat[0] = alpha[0]
            b_hat[0] = winv[0]

            for u in numba.prange(1, n):
                alpha_hat[u] = alpha[u] - beta[u - 1] * beta[u - 1] / alpha_hat[u - 1]
                b_hat[u] = winv[u] - beta[u - 1] * b_hat[u - 1] / alpha_hat[u - 1]

            xinv = np.ones(n, dtype=np.float64)
            xinv[-1] = b_hat[-1] / alpha_hat[-1]

            for u in range(n - 2, -1, -1):
                xinv[u] = (b_hat[u] - beta[u] * xinv[u + 1]) / alpha_hat[u]
            w[1:-1] = xinv

    # trasform heat equation back to original problem
    spot = np.zeros(m + 1, dtype=np.float64)
    v0 = np.zeros(m + 1, dtype=np.float64)
    for i in numba.prange(0, m + 1):
        x = a + i * dx
        spot[i] = strike * np.exp(x)
        v0[i] = strike * w[i] * np.exp(-(q - 1) * x / 2 - sigma ** 2 * mt / 2 * ((q - 1) ** 2 / 4 + q))

    return [v0, spot]


@numba.jit(nopython=True)
def fidi_bs_eu_numba2(strike, r, sigma, mt, a, b, m, nu_max, scheme):
    # strike: float or integer the strike price
    # r: float/integer the risk free interest rate
    # sigma: float/integer, volatility of the underlying
    # mt: float/integer, time until maturity in years
    # a/b: float/integer, bounds in space of the finite difference scheme
    # m: integer, grid in space
    # nu_max: integer, grid in time
    # scheme:  string corresponding to the fidi scheme: 'explicit', 'implicit' or 'cn' (cn is crank nicolson scheme)

    # returns list of v0 and spot
    # v0: option price values
    # spot: to v0 corresponding spot prices

    q = 2 * r / sigma ** 2
    dx = ((b - a) / m)
    dt = (sigma ** 2 * mt / (2 * nu_max))
    lamb = dt / dx ** 2

    # if explicit scheme chosen, check for stability
    if scheme == "explicit":
        if lamb >= 0.5:
            print("ERROR, Finite difference Scheme unstable: lambda >=0.5")
            return

    # initialize for a call option
    w = np.zeros(m + 1, dtype=np.float64)
    for i in range(0, m + 1):
        x = a + i * dx
        w[i] = np.maximum(np.exp(0.5 * x * (q + 1)) - np.exp(0.5 * x * (q - 1)), 0)
    w[0] = 0

    if scheme == "explicit":
        for i in range(1, nu_max+1):
            t = i * dt
            # explicit fidi scheme
            w[1:-1] = lamb * w[0:m - 1] + (1 - 2 * lamb) * w[1:m] + lamb * w[2:m + 1]
            # boundary condition of the call
            w[-1] = np.exp(0.5 * (q + 1) * b + (0.5 * (q + 1)) ** 2 * t) - np.exp(
                0.5 * (q - 1) * b + (0.5 * (q - 1)) ** 2 * t)

    elif scheme == "implicit":
        alpha = np.ones(m - 1, dtype=np.float64) * (1 + 2 * lamb)
        beta = np.ones(m - 2, dtype=np.float64) * (-lamb)
        for i in range(1, nu_max + 1):
            t = i * dt
            # boundary condition
            w[-1] = np.exp(0.5 * (q + 1) * b + (0.5 * (q + 1)) ** 2 * t) - np.exp(
                0.5 * (q - 1) * b + (0.5 * (q - 1)) ** 2 * t)
            w[-2] = w[-2] + lamb * w[-1]
            # implicit scheme
            w[1:-1] = tridiagonal_invers(alpha, beta, beta, w[1:-1])

    elif scheme == "cn":
        alpha = np.ones(m - 1, dtype=np.float64) * (1 + lamb)
        beta = np.ones(m - 2, dtype=np.float64) * (-0.5 * lamb)
        for i in range(1, nu_max + 1):
            t = i * dt
            # explicit part
            w[1:-1] = 0.5 * lamb * w[0:m - 1] + (1 - lamb) * w[1:m] + 0.5 * lamb * w[2:m + 1]
            # adjusting for boundary condition
            w[-1] = np.exp(0.5 * (q + 1) * b + (0.5 * (q + 1)) ** 2 * t) - np.exp(
                0.5 * (q - 1) * b + (0.5 * (q - 1)) ** 2 * t)
            w[-2] = w[-2] + 0.5 * lamb * w[-1]
            # implicit part
            w[1:-1] = tridiagonal_invers(alpha, beta, beta, w[1:-1])

    # trasform heat equation back to original problem
    spot = np.zeros(m + 1, dtype=np.float64)
    v0 = np.zeros(m + 1, dtype=np.float64)
    for i in range(0, m + 1):
        x = a + i * dx
        spot[i] = strike * np.exp(x)
        v0[i] = strike * w[i] * np.exp(-(q - 1) * x / 2 - sigma ** 2 * mt / 2 * ((q - 1) ** 2 / 4 + q))
    return [v0, spot]


@numba.jit(nopython=True)
def fidi_bs_american_numba(strike, r, sigma, mt, a, b, m, nu_max):
    # strike: float or integer the strike price
    # r: float/integer the risk free interest rate
    # sigma: float/integer, volatility of the underlying
    # mt: float/integer, time until maturity in years
    # a/b: float/integer, bounds in space of the finite difference scheme
    # m: integer, grid in space
    # nu_max: integer, grid in time

    # returns list of v0 and spot
    # v0: option price values
    # spot: to v0 corresponding spot prices
    # ------------------------------------------------------------------------------------------------------------------
    dx = (b - a) / m
    dt = sigma ** 2 * mt / (2 * nu_max)

    xtilde = a + np.arange(0, m + 1) * dx

    ttilde = np.arange(0, nu_max + 1) * dt

    lamb = dt / dx ** 2
    q = 2 * r / sigma ** 2

    # discretized exercise function for the put (for american Fidi scheme)
    w = np.maximum(np.exp(xtilde * (q - 1) / 2) - np.exp(xtilde * (q + 1) / 2), 0)

    # the tridiagonal Matrix with alpha on diagonal and beta to the right and gamma to the left
    alpha = np.ones(m - 1, dtype=np.float64) * (1 + lamb)
    beta = np.ones(m - 2, dtype=np.float64) * (-0.5 * lamb)
    n = len(alpha)

    bs = np.zeros(m - 1, np.float64)
    for i in range(0, nu_max):

        if i == round(nu_max * 0.5):
            print("Fidi progress: 50%")

        gnui = np.exp((q + 1) ** 2 * ttilde[i] / 4) * np.maximum(
            np.exp(xtilde * (q - 1) / 2) - np.exp(xtilde * (q + 1) / 2), 0)

        w[-1] = gnui[-1]
        w[0] = gnui[0]
        bs[1:-1] = w[2:-2] + 0.5 * lamb * (w[1:-3] - 2 * w[2:-2] + w[3:-1])

        bs[0] = w[1] + 0.5 * lamb * (w[2] - 2 * w[1] + gnui[0] + np.exp((q + 1) ** 2 * ttilde[i+1] / 4) * np.maximum(np.exp(a * (q - 1) / 2) - np.exp(a * (q + 1) / 2), 0))
        bs[-1] = w[-2] + 0.5 * lamb * (gnui[-1] - 2 * w[-2] + w[-3] + np.exp((q + 1) ** 2 * ttilde[i+1] / 4) * np.maximum(np.exp(b * (q - 1) / 2) - np.exp(b * (q + 1) / 2), 0))

        g_nui = gnui[1:-1]

        alpha_hat = np.zeros(n, dtype=np.float64)
        b_hat = np.zeros(n, dtype=np.float64)

        alpha_hat[-1] = alpha[-1]
        b_hat[-1] = bs[-1]

        for u in range(n - 2, -1, -1):
            alpha_hat[u] = alpha[u] - beta[u] * beta[u] / alpha_hat[u + 1]
            b_hat[u] = bs[u] - beta[u] * b_hat[u + 1] / alpha_hat[u + 1]

        xinv = np.zeros(n, dtype=np.float64)
        xinv[0] = np.maximum(b_hat[0] / alpha_hat[0], g_nui[0])
        for u in numba.prange(1, n):
            xinv[u] = np.maximum((b_hat[u] - beta[u - 1] * xinv[u - 1]) / alpha_hat[u], g_nui[u])

        w[1:-1] = xinv

    spot = np.zeros(m + 1, dtype=np.float64)
    v0 = np.zeros(m + 1, dtype=np.float64)
    for i in numba.prange(0, m + 1):
        x = a + i * dx
        spot[i] = strike * np.exp(x)
        v0[i] = strike * w[i] * np.exp(-(q - 1) * x / 2 - sigma ** 2 * mt / 2 * ((q - 1) ** 2 / 4 + q))
    return [v0, spot]
