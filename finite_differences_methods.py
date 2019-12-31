import numpy as np
import numba


@numba.jit(nopython=True, parallel=True)
def tridiagonal_invers(alpha, beta, gamma, b):
    # solution to Ax=b, where A is [n x n] tridiagonal matrix with values alpha, beta, gamma
    # alpha: [n x 1] numpy vector  of main diagonal of A
    # beta: [(n-1) x 1] numpy vector of diagonal above main diagonal (+1)
    # gamma: [(n-1) x 1] numpy vector of diagonal below main diagonal (-1)
    # b: [n x 1] numpy vector

    # returns x: [n x 1] numpy vector (solution to Ax=b)
    # ------------------------------------------------------------------------------------------------------------------
    n = len(alpha)
    alpha_hat = np.zeros(n, dtype=np.float64)
    b_hat = np.zeros(n, dtype=np.float64)

    alpha_hat[0] = alpha[0]
    b_hat[0] = b[0]

    for i in range(1, n):
        alpha_hat[i] = alpha[i] - gamma[i - 1] * beta[i - 1] / alpha_hat[i - 1]
        b_hat[i] = b[i] - gamma[i - 1] * b_hat[i - 1] / alpha_hat[i - 1]

    x = np.ones(n, dtype=np.float64)
    x[-1] = b_hat[-1] / alpha_hat[-1]

    for i in range(n - 2, -1, -1):
        x[i] = (b_hat[i] - beta[i] * x[i + 1]) / alpha_hat[i]
    return x


@numba.jit(nopython=True, parallel=True)
def brennon_schwartz_algorithm(alpha, beta, gamma, bs, g_nui):
    # Solution to Ax-b >= 0 ; x >= g and (Ax-b)'(x-g)=0 ; such that solution x satisfies x_i = g_i for i=1,...,k and x_i > g_i for i = k-1,...,n
    #  where A is tridiagonal with values alpha, beta, gamma
    # alpha: [n x 1] numpy vector  of main diagonal of A
    # beta: [(n-1) x 1] numpy vector of diagonal above main diagonal (+1)
    # gamma: [(n-1) x 1] numpy vector of diagonal below main diagonal (-1)
    # b: [n x 1] numpy vector

    # returns solution x: [n x 1] numpy vector
    # ------------------------------------------------------------------------------------------------------------------
    n = len(alpha)
    alpha_hat = np.zeros(n, dtype=np.float64)
    b_hat = np.zeros(n, dtype=np.float64)

    alpha_hat[-1] = alpha[-1]
    b_hat[-1] = bs[-1]

    for i in range(n - 2, -1, -1):
        alpha_hat[i] = alpha[i] - beta[i] * gamma[i] / alpha_hat[i + 1]
        b_hat[i] = bs[i] - beta[i] * b_hat[i + 1] / alpha_hat[i + 1]

    x = np.zeros(n, dtype=np.float64)
    x[0] = np.maximum(b_hat[0] / alpha_hat[0], g_nui[0])
    for i in range(1, n):
        x[i] = np.maximum((b_hat[i] - gamma[i - 1] * x[i - 1]) / alpha_hat[i], g_nui[i])
    return x
