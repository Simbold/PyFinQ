## import pricing functions
import numpy as np
from pricing.vanilla.binomial_tree import binomial_tree_bs
from pricing.vanilla.finite_differences import fidi_bs_eu, fidi_bs_american

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
m = 500
# number of steps in time
nu_max = 10000

[v0_fidi, s0_fidi] = fidi_bs_american(strike, r, sigma, mt, a, b, m, nu_max)

# list of spot prices
spot = [26, 28, 30, 32, 34, 36, 38]
fidi_result = np.interp(spot, s0_fidi, v0_fidi)  # find the price for different spots via linear interpolation
print(fidi_result)
