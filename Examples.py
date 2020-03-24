## import pricing functions
import numpy as np

from monte_carlo import sim_gbm_paths
from pricing.vanilla.binomial_tree import binomial_tree_bs
from pricing.vanilla.finite_differences import fidi_bs_eu, fidi_bs_american
from pricing.vanilla.integration import closed_form_bs_eu, laplace_heston_eu, fast_fourier_bs_eu, fast_fourier_heston_eu
from pricing.vanilla.monte_carlo import monte_carlo_bs_eu, monte_carlo_bs_am

## Binomial tree: Pricing via the Binomial tree model of Cox Ross and Rubinstein

spot = 32  # spot price
strike = 32  # strike price
r = 0.02  # annual risk free interest rate
sigma = 0.3  # volatility
mt = 1.4  # maturity time in years
m = 10000  # number of steps, must be an integer
american_exercise = True  # American exercise True or False depending on options exercise style American/European style
option_type = "put"  # option_type can be "put" or "call"

# Compute:
v0_binomial_tree = binomial_tree_bs(spot, strike, r, sigma, mt, m, option_type, american_exercise)
print(v0_binomial_tree)

## Finite-differences European exercise call

strike = 32  # strike price
r = 0.02  # annual risk free interest rate
sigma = 0.3  # volatility
mt = 1.4  # maturity time in years

# bounds in space
a = -0.5
b = 0.5

m = 500  # number of steps in space, must be an integer
nu_max = 10000  # number of steps in time

# type of finite differences scheme (explicit, implicit or Crank-Nicolson (cn))
scheme = "cn"  # possible are "cn", "implicit", "explicit"

#  Compute:
[v0_fidi_eu, s0_fidi_eu] = fidi_bs_eu(strike, r, sigma, mt, a, b, m, nu_max, scheme)

spot = [26, 28, 30, 32, 34, 36, 38]  # list of spot prices
v0_fidi_eu_interp = np.interp(spot, s0_fidi_eu, v0_fidi_eu)  # find the price for different spots via linear interpolation
print(v0_fidi_eu_interp)

## Finite-differences American exercise put

strike = 32  # strike price
r = 0.02  # annual risk free interest rate
sigma = 0.3  # volatility
mt = 1.4  # maturity time in years

# bounds in space
a = -0.5
b = 0.5

m = 1000  # number of steps in space, must be an integer
nu_max = 20000  # number of steps in time

# Compute:
[v0_fidi_am, s0_fidi_am] = fidi_bs_american(strike, r, sigma, mt, a, b, m, nu_max)

spot = [26, 28, 30, 32, 34, 36, 38]  # list of spot prices
v0_fidi_am_interp = np.interp(spot, s0_fidi_am, v0_fidi_am)  # find the price for different spots via linear interpolation
print(v0_fidi_am_interp)

## black scholes explicit formula

spot = 110  # spot price
strike = 100  # strike price
r = 0.05  # annual risk free interest rate
sigma = 0.2  # volatility
mt = 1  # maturity time in years
option_type = "call"  # put or call

# Compute:
vt_bs = closed_form_bs_eu(spot, strike, r, sigma, mt, option_type, t=0)
print(vt_bs)

## heston call/put via laplace transform

spot = 110  # spot price
strike = 100  # strike price
r = 0.05  # annual risk free interest rate
mt = 1  # maturity time in years

# heston volatility dynamics parameters
sigma_tilde = 0.2
nu0 = 0.3**2
kappa = 0.3**2
lamb = 2.5

option_type = "call"  # put or call

[vt_heston, abserr] = laplace_heston_eu(spot, strike, r, sigma_tilde, mt, nu0, kappa, lamb, option_type, t=0)
print(vt_heston)

## Pricing via Fast Fourier transform in the black-scholes model

# Note:
# FFT prices simultaneously for a list of strikes
# After FFT pricing, returned values are obtained via linear interpolation
# m/n is the mesh size of the integral approximation via the midpoint rule, m should be large and m/n should be small
# however m/n also effects the gap size between values over which is interpolated and should not be too small

spot = 110
strikes = [95, 100, 105, 110, 115, 120, 125]  # a list of strike prices
# strikes = np.arange(90, 180, 0.1)  # or a numpy array of strike prices
r = 0.05
sigma = 0.2
mt = 1
option_type = "call"

# Compute:
[vt_fft_bs_interpolated, vt_fft_bs, strikes_fft_bs] = fast_fourier_bs_eu(spot, strikes, r, sigma, mt, option_type, n=10000, m=400, t=0)
print(vt_fft_bs_interpolated)
print(vt_fft_bs)
print(strikes_fft_bs)



### Pricing via Fast Fourier transform in the heston model

# Note:
# FFT prices simultaneously for a list of strikes
# After FFT pricing, returned values are obtained via linear interpolation
# m/n is the mesh size of the integral approximation via the midpoint rule, m should be large and m/n should be small
# m also effects the gap size between values over which is interpolated and should not be too small

spot = 110
strikes = [95, 100, 105, 110, 115, 120, 125]  # a list of strike prices
# strikes = np.arange(90, 180, 0.1)  # or a numpy array of strike prices
r = 0.05
mt = 1
option_type = "call"

sigma_tilde = 0.2
nu0 = 0.3**2
kappa = 0.3**2
lamb = 2.5

[vt_fft_heston_interpolated, vt_fft_heston, strikes_fft_heston] = fast_fourier_heston_eu(spot, strikes, r, sigma_tilde, mt, nu0, kappa, lamb, option_type, n=10000, m=400, t=0)
print(vt_fft_heston_interpolated)
print(vt_fft_heston)
print(strikes_fft_heston)

## Monte Carlo Black Scholes EU

spot = 110  # spot price
strike = 100  # strike price
r = 0.05  # annual risk free interest rate
sigma = 0.2  # volatility
mt = 1  # maturity time in years
d = 0 # dividend yield, NOTE: only possible without importance sampling
option_type = "put"
antithetic = True
n = 100000

# reference value put: 2.785896190661841

[v0, ci] = monte_carlo_bs_eu(spot, strike, r, d, sigma, mt, n, option_type, antithetic)
print(v0)
print(ci)

# reference value call: 17.66295374059044
option_type = "call"
[v0, ci] = monte_carlo_bs_eu(spot, strike, r, d, sigma, mt, n, option_type, antithetic)
print(v0)
print(ci)  # 95% confidence interval


## Monte carlo Black Scholes EU Importance Sampling

spot = 110
strike = 60
r = 0.05
sigma = 0.2
mt = 1
n = 100000
d = 1  # dividend yield, NOTE: must be zero or otherwise will be set to zero if importance sampling is used
antithetic = True
#  reference value: 0.002160509264695208
importance_sampling = True
option_type = "put"
[v0, ci] = monte_carlo_bs_eu(spot, strike, r, d, sigma, mt, n, option_type, antithetic, importance_sampling)
print(v0)
print(ci)

spot = 110
strike = 180
#  reference value: 0.12896384364721736
option_type = "call"
[v0, ci] = monte_carlo_bs_eu(spot, strike, r, d, sigma, mt, n, option_type, antithetic, importance_sampling)
print(v0)
print(ci)


## Monte Carlo Black Scholes Am using longstaff schwartz


spot = 32  # spot price
strike = 32  # strike price
r = 0.02  # annual risk free interest rate
sigma = 0.3  # volatility
mt = 1.4  # maturity time in years
d = 0  # annual dividend yield
m = 100  # number of equidistant exercise dates
n = 100000  # number of simulated paths
antithetic = True  # True if half of the simulated paths should be antithetic
option_type = "put"  # put or call
k = 3  # number of basis function to use, for polynomial no limit, for laguerre 7 is the maximum
basis = "laguerre"  # polynomial and laguerre basis functions are possible, laguerre basis has
fit_method = "inv"  # possible values are 'inv' (usual inverse method), 'qr' (QR-decomposition), 'svd' singular value decomposition
# reference from binomiall tree: 4.1002953921226295

# generate Geometric Brownian Motion paths
paths = sim_gbm_paths(spot, sigma, mt, r, m, n, d, antithetic)

[v0, se] = monte_carlo_bs_am(strike, r, mt, option_type, paths, k, basis, fit_method)
print(v0)
print(se)  # the standard error of the Monte Carlo estimate
