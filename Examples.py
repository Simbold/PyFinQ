## binomial tree
from pricing.vanilla.binomial_tree import binomial_tree_bs

# spot price
spot = 33
# risk free interest rate
r = 0.02
# volatility
sigma = 0.3
# maturity time in years
mt = 1.4
# number of steps, must be an integer
m = 10000
# strike price
strike = 33
# American exercise True or False depending on options exercise style American/European style
american_exercise = True
# option_type can be "put" or "call"
option_type = "put"


v0 = binomial_tree_bs(spot, strike, r, sigma, mt, m, option_type, american_exercise)
print(v0)

##