## import pricing functions
import numpy as np
from pricing.vanilla.binomial_tree import binomial_tree_bs
from pricing.vanilla.finite_differences import fidi_bs_eu, fidi_bs_american
from pricing.vanilla.integration import closed_form_bs_eu, integrate_heston_eu

## binomial tree

# spot price

spot = 32
# risk free interest rate
r = 0.02
# volatility
sigma = 0.3
# maturity time in years
mt = 1.4
# number of steps, must be an integer
m = 10000
# strike price
strike = 32
# American exercise True or False depending on options exercise style American/European style
american_exercise = True
# option_type can be "put" or "call"
option_type = "put"


v0 = binomial_tree_bs(spot, strike, r, sigma, mt, m, option_type, american_exercise)
print(v0)

## Finite-differences European exercise

# strike price
strike = 32
# risk free interest rate
r = 0.02
# volatility
sigma = 0.3
# maturity time in years
mt = 1.4

# bounds in space
a = -0.5
b = 0.5
# number of steps in space, must be an integer
m = 500
# number of steps in time
nu_max = 10000
# type of finite differences scheme (explicit, implicit or Crank-Nicolson (cn))
scheme = "cn"  # possible are "cn", "implicit", "explicit"

[v0_fidi, s0_fidi] = fidi_bs_eu(strike, r, sigma, mt, a, b, m, nu_max, scheme)

# list of spot prices
spot = [26, 28, 30, 32, 34, 36, 38]
fidi_result = np.interp(spot, s0_fidi, v0_fidi)  # find the price for different spots via linear interpolation
print(fidi_result)

## Finite-differences American exercise

# strike price
strike = 32
# risk free interest rate
r = 0.02
# volatility
sigma = 0.3
# maturity time in years
mt = 1.4

# bounds in space
a = -0.5
b = 0.5
# number of steps in space, must be an integer
m = 1000
# number of steps in time
nu_max = 20000

[v0_fidi, s0_fidi] = fidi_bs_american(strike, r, sigma, mt, a, b, m, nu_max)

# list of spot prices
spot = [26, 28, 30, 32, 34, 36, 38]
fidi_result = np.interp(spot, s0_fidi, v0_fidi)  # find the price for different spots via linear interpolation
print(fidi_result)

## black scholes explicit formula

spot = 110
strike = 100
r = 0.05
sigma = 0.2
mt = 1
option_type = "put"

result_bs = closed_form_bs_eu(spot, strike, r, sigma, mt, option_type, t=0)
print(result_bs)

## heston call/put via laplace transform

spot = 110
strike = 100
r = 0.05
mt = 1
# heston volatility dynamics parameters
sigma_tilde = 0.2
nu0 = 0.3**2
kappa = 0.3**2
lamb = 2.5

option_type = "put"

[v0_heston, abserr] = integrate_heston_eu(spot, strike, r, sigma_tilde, mt, nu0, kappa, lamb, option_type, t=0)
print(v0_heston)
