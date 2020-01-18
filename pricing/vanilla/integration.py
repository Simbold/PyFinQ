import numpy as np
from scipy import stats, integrate
from integration_functions import laplace_transform_vanilla, heston_characteristic_function, \
    black_scholes_characteristic_function


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


def laplace_heston_eu(spot, strike, r, sigma_tilde, mt, nu0, kappa, lamb, option_type="call", t=0):
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


def laplace_transform_vanilla_0(z):
    # laplace transform for vanilla option with strike=exp(kappa)
    # for for pricing via fast fourier transform
    return 1/((z-1)*z)


def fast_fourier_bs_eu(spot, strikes, r, sigma, mt, option_type="call", n=10000, m=400, t=0):
    # spot: underlying spot price
    # strikes: strike prices can be a list of multiple strikes
    # r: rik free interest rate
    # sigma: volatility
    # mt: time to maturity in years
    # option_type: Type of the option either call or put
    # n, m: delta=m/n is the mesh size of the integral approximation via the midpoint rule, m should be large and m/n should be small
    # t: time at which to evaluate the option

    # returns a list of three values:
    # 1. option values corresponding to the inputed strikes
    # 2. options values without interpolation
    # 3. strikes corresponding to the prices without interpolation
    # ------------------------------------------------------------------------------------------------------------------
    if option_type == "call":
        R = 1.1
    elif option_type == "put":
        R = -0.1
    else:
        print("ERROR, option_type must be either 'call', or 'put'")
        return

    def g(u):
        return laplace_transform_vanilla_0(R+1j*u)*black_scholes_characteristic_function(u-1j*R, np.log(spot), r, sigma, mt, t)

    delta = m/n
    kappa1 = np.log(np.min(strikes))

    x = np.zeros(n, dtype=np.complex128)
    for i in range(1, n):
        x[i-1] = g((i - 0.5)*delta) * delta * np.exp(-1j*(i-1)*delta*kappa1)

    # perform DFT using the efficient FFT algorithm
    x_hat = np.fft.fft(x)
    # compute vector kappa
    kappa_m = kappa1 + (np.arange(1, n+1) - 1) * 2 * np.pi / m

    strikes_fft = np.exp(kappa_m)
    # finally compute the option prices
    vt_fft = (np.exp(-r*(mt-t) + (1-R)*kappa_m))/np.pi * np.real(x_hat*np.exp(-1j * delta * kappa_m / 2))
    # interpolate strike values
    vt_fft_interpolated = np.interp(strikes, strikes_fft, vt_fft)
    return [vt_fft_interpolated, vt_fft, strikes_fft]


def fast_fourier_heston_eu(spot, strikes, r, sigma_tilde, mt, nu0, kappa, lamb, option_type="call", n=10000, m=400, t=0):
    # spot: underlying spot price
    # strikes: strike prices can be a list of multiple strikes
    # r: rik free interest rate
    # sigma_tilde, nu0, kappa, lambda: heston volatility dynamic parameters
    # mt: time to maturity in years
    # option_type: Type of the option either call or put
    # n, m: delta=m/n is the mesh size of the integral approximation via the midpoint rule, m should be large and m/n should be small
    # t: time at which to evaluate the option

    # returns a list of three values:
    # 1. option values corresponding to the inputed strikes
    # 2. options values without interpolation
    # 3. strikes corresponding to the prices without interpolation
    # ------------------------------------------------------------------------------------------------------------------
    if option_type == "call":
        R = 1.1
    elif option_type == "put":
        R = -0.1
    else:
        print("ERROR, option_type must be either 'call', or 'put'")
        return

    def g(u):
        return laplace_transform_vanilla_0(R+1j*u)*heston_characteristic_function(u-1j*R,  np.log(spot), nu0, r, sigma_tilde, mt, t, kappa, lamb)

    delta = m/n
    kappa1 = np.log(np.min(strikes))

    x = np.zeros(n, dtype=np.complex128)
    for i in range(1, n):
        x[i-1] = g((i - 0.5)*delta) * delta * np.exp(-1j*(i-1)*delta*kappa1)

    # perform DFT using the efficient FFT algorithm
    x_hat = np.fft.fft(x)
    # compute vector kappa
    kappa_m = kappa1 + (np.arange(1, n+1) - 1) * 2 * np.pi / m

    strikes_fft = np.exp(kappa_m)
    # finally compute the option prices
    vt_fft = (np.exp(-r*(mt-t) + (1-R)*kappa_m))/np.pi * np.real(x_hat*np.exp(-1j * delta * kappa_m / 2))
    # interpolate strike values
    vt_fft_interpolated = np.interp(strikes, strikes_fft, vt_fft)
    return [vt_fft_interpolated, vt_fft, strikes_fft]

