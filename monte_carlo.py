import numpy as np
import numba


@numba.jit(nopython=True, parallel=True)
def sim_gbm_paths(spot, sigma, mt, r, m, n, d, antithetic=True):
    # m number of time steps
    # n number of paths
    # r the risk free interest rate
    # T maturity time

    # returns a [n x (m+1)] numpy array of simulated paths

    n2 = int(n / 2)
    n1 = n2 * 2
    paths = np.zeros((n1, m + 1), dtype=np.float64)
    paths[:, 0] = spot
    dt = mt / m
    if antithetic:
        z = np.random.normal(loc=0, scale=1, size=(n2, m))
        for i in numba.prange(0, n2):
            paths[i, 1:m + 1] = spot * np.exp(
                np.cumsum((((r - d) - 0.5 * sigma ** 2) * dt) + (sigma * np.sqrt(dt) * z[i, :])))
            paths[i + n2, 1:m + 1] = spot * np.exp(
                np.cumsum((((r - d) - 0.5 * sigma ** 2) * dt) + (sigma * np.sqrt(dt) * -z[i, :])))

    elif not antithetic:
        z = np.random.normal(loc=0, scale=1, size=(n1, m))
        for i in numba.prange(0, n1):
            paths[i, 1:m + 1] = spot * np.exp(
                np.cumsum((((r - d) - 0.5 * sigma ** 2) * dt) + (sigma * np.sqrt(dt) * z[i, :])))
    else:
        print("ERROR: antithetic must be either True or False")
        return
    return paths


@numba.jit(nopython=True, parallel=True)
def mc_bs_eu_numba(spot, strike, r, d, sigma, mt, n, payoff, antithetic):
    n2 = np.int64(n / 2)
    n1 = n
    if antithetic:
        z = np.zeros(n1, dtype=np.float64)
        z[0:n2] = np.random.normal(loc=0, scale=1, size=n2)
        z[n2:(n1 + 1)] = - z[0:n2]
        paths = np.exp(-r * mt) * payoff(spot * np.exp(((r-d) - 0.5 * sigma ** 2) * mt + sigma * np.sqrt(mt) * z), strike)
    elif not antithetic:
        z = np.random.normal(loc=0, scale=1, size=n1)
        paths = np.exp(-r * mt) * payoff(spot * np.exp(((r-d) - 0.5 * sigma ** 2) * mt + sigma * np.sqrt(mt) * z), strike)
    else:
        print("ERROR: antithetic must be either True or False")
        return
    v0 = np.mean(paths)
    var = np.var(paths)
    return [v0, var]


@numba.jit(nopython=True, parallel=True)
def mc_bs_eu_is_numba(spot, strike, r, sigma, mt, mu, n, payoff, antithetic):
    n2 = int(n / 2)
    n1 = n2 * 2
    if antithetic:
        y = np.zeros(n1, dtype=np.float64)
        y[0:n2] = np.random.normal(loc=0, scale=1, size=n2)
        y[n2:(n1 + 1)] = - y[0:n2]
        y = y + mu
        paths = np.exp(-r * mt - y * mu + 0.5 * mu ** 2) * payoff(
            spot * np.exp((r - 0.5 * sigma ** 2) * mt + sigma * np.sqrt(mt) * y), strike)
    elif not antithetic:
        y = np.random.normal(loc=mu, scale=1, size=n1)
        paths = np.exp(-r * mt - y * mu + 0.5 * mu ** 2) * payoff(
            spot * np.exp((r - 0.5 * sigma ** 2) * mt + sigma * np.sqrt(mt) * y), strike)
    else:
        print("ERROR: antithetic must be either True or False")
        return

    v_is = np.mean(paths)
    var = np.var(paths)
    return [v_is, var]


@numba.jit(nopython=True, parallel=True)
def lr_qr(X, y):
    q, r = np.linalg.qr(X)
    beta = np.linalg.solve(r, np.dot(q.T, y))
    return beta


@numba.jit(nopython=True, parallel=True)
def lr(X, y):
    beta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
    return beta


# coefficients using the singular value decomposition
@numba.jit(nopython=True, parallel=True)
def lr_svd(X, y):
    u, s, v = np.linalg.svd(np.dot(X.T, X))
    pseudo_inv = np.dot(np.transpose(v) @ np.diag(1 / s) @ np.transpose(u), X.T)
    beta = np.dot(pseudo_inv, y)
    return beta


def longstaff_schwartz(strike, mt, r, paths, k, norm_factor, payoff, basis_function, fit_method="qr", itm=True):
    # paths must be a [n x (m+1)] numpy array of the simulated price process with #n paths and #m time steps
    # fit_method are: lr_qr, lr_inv and lr_svd
    # basis_function values: laguerre_basis and poly_basis.
    # mt maturity time
    # k number of basis functions excluding the intercept
    # coefficients: either False or True depending if matrix of estimated coefficients shall be returned or not
    # r risk free rate

    # returns a list consisting of:
    # v0 price of the option
    # corresponding standard error
    # if coefficients=True: a numpy array of coefficients
    m = paths.shape[1] - 1
    n = paths.shape[0]
    dt = mt / m
    v = payoff(paths[:, -1], strike) * np.exp(-r * dt)

    for t in range(m - 1, 0, -1):  # loop from time m-1 backwards to time 1 to recursivly compute v0

        if itm:
            itm = np.where(payoff(paths[:, t], strike) > 0)
        elif not itm:
            itm = np.where(payoff(paths[:, t], strike) >= 0)

        if len(itm[0]) > 0:  # if no paths are in the money regression is skipped
            if len(itm[0]) < (k + 1):  # if regression possible with only in the money paths, otherwise use all paths
                itm = np.where(paths[:, t])
                print("all paths used")

            exercise = payoff(paths[:, t], strike)
            A = basis_function(paths[itm, t] / norm_factor, k, strike)
            if fit_method == "qr":
                beta = lr_qr(A, v[itm] / norm_factor)
            elif fit_method == "svd":
                beta = lr_svd(A, v[itm] / norm_factor)
            elif fit_method == "inv":
                beta = lr(A, v[itm] / norm_factor)
            else:
                print("ERROR: unkown method! possible methods are: 'qr' , 'inv' and 'svd'")
                return

            cv = np.zeros(n)
            cv[itm] = np.dot(A, beta)
            eex = np.where(
                (exercise / norm_factor >= cv) & (exercise > 0))  # indices of optimal exercise for in the money paths
            v[eex] = exercise[eex]

        v = v * np.exp(-r * dt)  # discount one step for next iteration

    v0 = np.maximum(np.mean(v), payoff(paths[0, 0], strike))
    se = np.sqrt(n / (n - 1) * np.var(v) / n)  # compute the standard error of the simulation
    return [v0, se]