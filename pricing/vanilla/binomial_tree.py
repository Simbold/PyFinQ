import numpy as np
import numba
from tree_methods import generate_tree_state, generate_full_tree, european_state_iterator, conditional_state_iterator, \
    american_put_condition


def binomial_tree_bs(spot, strike, r, sigma, mt, m, option_type="call", american_exercise=False):
    # spot: underlying spot price
    # strike: strike price
    # r: rik free interest rate
    # sigma: volatility of the underlying
    # mt: time to maturity in years
    # m: integer number of steps
    # option_type: Type of the option either call or put
    # american_exercise: either False if exercise style is European or True if exercise style is American

    # returns the cox rox rubinstein option value
    # ------------------------------------------------------------------------------------------------------------------
    # check input
    if round(m) == m:
        pass
    else:
        print("ERROR: m must be interpretable as integer")
        return
    if ((option_type == "call") | (option_type == "put")) & ((american_exercise == False) | (american_exercise == True)):
        # if input correct compute output
        v0 = binomial_tree_bs_numba2(spot, strike, r, sigma, mt, m, option_type, american_exercise)
    else:
        print("ERROR: variable: american must be True or False; variable: option_type must be 'call' or 'put'")
        return
    return v0


@numba.jit(nopython=True, parallel=True)
def binomial_tree_bs_numba(spot, strike, r, sigma, mt, m, option_type, american):
    # spot: underlying spot price
    # strike: strike price
    # r: rik free interest rate
    # sigma: volatility of the underlying
    # mt: time to maturity in years
    # m: number of steps
    # otype: type of the option either call or put
    # american: either False if exercise style is european or True if exercise style is American

    # returns the cox rox rubinstein option value
    # ------------------------------------------------------------------------------------------------------------------
    # calculate cox ross rubinstein parameters u, d, q, dt
    dt = mt / m
    b = 0.5 * (np.exp(-r * dt) + np.exp((r + sigma ** 2) * dt))
    u = b + np.sqrt(b ** 2 - 1)
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    if (option_type == "call") & ((american == False) | (american == True)):
        # generate tree state at maturity
        final_tree_state = generate_tree_state(spot, m, u, d)
        # compute payoff to initialize value process
        v = np.maximum(final_tree_state - strike, 0)
        # compute option value
        v0 = european_state_iterator(v, m, q, r, dt)
    elif (american == False) & (option_type == "put"):
        # generate tree state at maturity
        final_tree_state = generate_tree_state(spot, m, u, d)
        # compute payoff to initialize value process
        v = np.maximum(strike - final_tree_state, 0)
        # compute option value
        v0 = european_state_iterator(v, m, q, r, dt)
    elif (american == True) & (option_type == "put"):
        # generate entire tree
        full_tree = generate_full_tree(spot, m, u, d)
        # compute payoff to initialize value process
        v = np.maximum(strike - full_tree[:, -1], 0)
        # compute option value
        v0 = conditional_state_iterator(v, full_tree, m, q, r, dt, american_put_condition, strike)
    else:
        v0 = -1
    return v0


@numba.jit(nopython=True, parallel=True)
def binomial_tree_bs_numba2(spot, strike, r, sigma, mt, m, option_type="call", american_exercise=False):
    # spot: underlying spot price
    # strike: strike price
    # r: rik free interest rate
    # sigma: volatility of the underlying
    # mt: time to maturity in years
    # m: number of steps
    # option_type: type of the option either 'call' or 'put'
    # american_exercise: either False if exercise style is european or True if exercise style is American

    # returns the cox rox rubinstein option value
    # ------------------------------------------------------------------------------------------------------------------
    # calculate cox ross rubinstein parameters u, d, q, dt
    dt = mt / m
    b = 0.5 * (np.exp(-r * dt) + np.exp((r + sigma ** 2) * dt))
    u = b + np.sqrt(b ** 2 - 1)
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)

    if option_type == "call":
        # generate tree state at maturity
        tree = np.zeros(m + 1)
        tree[0] = spot
        print("generate tree state at maturity")
        for j in range(1, m + 1):
            tree[j] = tree[j - 1] * d
            tree[0:j] = tree[0:j] * u

        # compute payoff to initialize value process
        v = np.maximum(tree - strike, 0)
        # compute option value
        for j in range(m, 0, -1):
            v = (q * v[0:j] + (1 - q) * v[1:j + 1]) * np.exp(-r * dt)
            if j == round(m * 0.5):
                print("Backwards iteration at: 50%")
        v0 = v[0]

    elif (american_exercise == False) & (option_type == "put"):
        # generate tree state at maturity
        tree = np.zeros(m + 1)
        tree[0] = spot
        print("generate tree state at maturity")
        for j in range(1, m + 1):
            tree[j] = tree[j - 1] * d
            tree[0:j] = tree[0:j] * u
        # compute payoff to initialize value process
        v = np.maximum(strike - tree, 0)
        # compute option value
        for j in range(m, 0, -1):
            v = (q * v[0:j] + (1 - q) * v[1:j + 1]) * np.exp(-r * dt)
            if j == round(m * 0.5):
                print("Backwards iteration at: 50%")
        v0 = v[0]

    elif (american_exercise == True) & (option_type == "put"):
        # generate entire tree
        full_tree = np.zeros((m + 1, m + 1))
        full_tree[0, 0] = spot
        print("generating tree")
        for j in range(1, m + 1):
            full_tree[j, j] = full_tree[j - 1, j - 1] * d
            full_tree[0:j, j] = full_tree[0:j, j - 1] * u

        # compute payoff to initialize value process
        v = np.maximum(strike - full_tree[:, -1], 0)
        # compute option value
        for j in range(m, -1, -1):
            for i in range(0, j):
                v[i] = np.maximum((q*v[i]+(1-q)*v[i+1])*np.exp(-r*dt), np.maximum(strike-full_tree[i, j-1], 0))
                if (j == round(m * 0.75)) & (i == 0):
                    print("Backwards iteration at: 50%")
            v = v[0:j]
        v0 = v[0]

    else:
        v0 = -1
    return v0



