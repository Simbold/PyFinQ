import numpy as np
from scipy import stats, integrate
from pricing.integration_functions import heston_characteristic_function, laplace_transform_vanilla


def closed_form_bs_eu(spot, strike, r, sigma, mt, option_type="call", t=0):
    # spot: underlying spot price
    # strike: strike price
    # r: rik free interest rate
    # sigma: volatility of the underlying
    # mt: time to maturity in years
    # option_type: Type of the option either call or put
    # t: time at which to evaluate the option

    # returns the European vanilla option value via the Black-Scholes closed form formula
    # ------------------------------------------------------------------------------------------------------------------

    d1 = (np.log(spot / strike) + (r + 0.5 * (sigma ** 2)) * (mt - t)) / sigma * np.sqrt(mt - t)
    d2 = d1 - sigma * np.sqrt(mt - t)

    if option_type == "call":
        v_t = spot * stats.norm.cdf(d1) - strike * np.exp(-r * (mt - t)) * stats.norm.cdf(d2)
    elif option_type == "put":
        v_t = strike * np.exp(-r * (mt - t)) * stats.norm.cdf(-d2) - spot * stats.norm.cdf(-d1)
    else:
        print("ERROR: option_type must be 'call' or 'put'")
        return
    return v_t


def integrate_heston_eu(spot, strike, r, sigma_tilde, mt, nu0, kappa, lamb, option_type="call", t=0):
    # spot: underlying spot price
    # strike: strike price
    # r: rik free interest rate
    # sigma_tilde, nu0, kappa, lambda: heston volatility dynamic parameters
    # mt: time to maturity in years
    # option_type: Type of the option either call or put
    # t: time at which to evaluate the option

    # returns list of two values
    # v_t: the European vanilla option value in the heston model via laplace transform
    # abserr: estimate of the absolute error of the numerical integration scheme
    # ------------------------------------------------------------------------------------------------------------------

    if option_type == "call":
        R = 2
    elif option_type == "put":
        R = -1
    else:
        print("ERROR, option_type must be either 'call', or 'put'")
        return

    def heston_call_integrant(u):
        y = (np.exp(-r*mt)/np.pi)*np.real(laplace_transform_vanilla(u*1j+R, strike)*heston_characteristic_function(u-1j*R,  np.log(spot), nu0, r, sigma_tilde, mt, t, kappa, lamb))
        return y

    [v0_heston, abserr] = integrate.quad(heston_call_integrant, 0, 100)[0:2]
    return [v0_heston, abserr]
