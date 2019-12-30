import numpy as np
import numba


@numba.jit(nopython=True, parallel=True)
def generate_full_tree(spot, m, u, d):
    # spot: underlyings spot price
    # m: integer number of steps
    # u: up value
    # d: down value

    # returns a full price tree as [(m+1)x(m+1)] numpy array

    tree = np.zeros((m+1, m+1))
    tree[0, 0] = spot
    print("generating tree")
    for j in range(1, m+1):
        tree[j, j] = tree[j-1, j-1] * d
        tree[0:j, j] = tree[0:j, j-1] * u
    return tree


@numba.jit(nopython=True, parallel=True)
def generate_tree_state(spot, m, u, d):
    # spot: underlyings spot price
    # m: integer number of steps
    # u: up value
    # d: down value

    # returns tree state at maturity as [(m+1),] array

    tree = np.zeros(m + 1)
    tree[0] = spot
    print("generate tree state at maturity")
    for j in range(1, m + 1):
        tree[j] = tree[j-1] * d
        tree[0:j] = tree[0:j] * u
    return tree


@numba.jit(nopython=True, parallel=True)
def european_state_iterator(v, m, q, r, dt):
    # v: an [(m+1),] array of the option value state at maturity
    # m: integer number of steps
    # q: Q probability of up
    # r: risk free rate
    # dt: mT/m

    # returns value of the option

    for j in range(m, 0, -1):
        v = (q * v[0:j] + (1 - q) * v[1:j + 1]) * np.exp(-r * dt)
        if j == round(m * 0.5):
            print("Backwards iteration at: 50%")
    return v[0]


@numba.jit(nopython=True, parallel=True)
def european_full_iterator(v, m, q, r, dt):
    # v: an initial [(m+1)x(m+1)] numpy array of the option value  with v[:, -1] option value state at maturity
    # m: integer number of steps
    # q: Q probability of up
    # r: risk free rate
    # dt: mT/m

    # returns [(m+1)x(m+1)] numpy array of full value tree of the option

    for j in range(m, 0, -1):
        v[0:j, j - 1] = (q * v[0:j, j] + (1 - q) * v[1:j + 1, j]) * np.exp(-r * dt)
        if j == round(m * 0.5):
            print("Backwards iteration at: 50%")
    return v[0]


@numba.jit(nopython=True)
def conditional_state_iterator(v, tree, m, q, r, dt, condition, cond1=0, cond2=0, cond3=0):
    # v: an [(m+1),] array of the option value state at maturity    # m: number of steps
    # tree: tree state at maturity as [(m+1),] numpy array
    # m: integer number of steps
    # q: Q probability of up
    # r: risk free rate
    # dt: mT/m
    # condition: condition function on how to manipulate state (for example american early exercise or barrier etc.)

    # returns value of the option

    for j in range(m, -1, -1):
        for i in range(0, j):
            v[i] = condition((q * v[i] + (1 - q) * v[i + 1]) * np.exp(-r * dt), tree[i, j - 1], cond1, cond2, cond3)
            if (j == round(m*0.75)) & (i == 0):
                print("Backwards iteration at: 50%")
        v = v[0:j]
    return v[0]


@numba.jit(nopython=True, parallel=True)
def conditional_full_iterator(v, tree, m, q, r, dt, condition, cond1, cond2, cond3):
    # v: an initial [(m+1)x(m+1)] numpy array of the option value  with v[:, -1] option value state at maturity
    # m: integer number of steps
    # tree: full price tree as [(m+1)x(m+1)] numpy array
    # q: Q probability of up
    # r: risk free rate
    # dt: mT/m
    # condition: condition function on how to manipulate state (for example american early exercise or barrier etc.)

    # returns [(m+1)x(m+1)] numpy array of full value tree of the option

    for j in range(m, -1, -1):
        for i in numba.prange(0, j):
            v[i, j - 1] = condition((q * v[i, j] + (1 - q) * v[i + 1, j]) * np.exp(-r * dt), tree[i, j - 1], cond1, cond2, cond3)
            if (j == round(m*0.75)) & (i == 0):
                print("Backwards iteration at: 50%")
    return v[0]


@numba.jit(nopython=True)
def american_put_condition(v, s, strike, cond2=0, cond3=0):
    # function to be passed to conditional iterator for american put
    vm = np.maximum(v, np.maximum(strike - s, 0))
    return vm

