import numpy as np
from algo import stoch_price
# Helper function to calculate the finite difference derivative
def finite_difference(func, var, h=1e-5, *args):
    args_up = list(args)
    args_down = list(args)
    args_up[var] += h
    args_down[var] -= h
    return (func(*args_up) - func(*args_down)) / (2 * h)

# Wrapper function to calculate the price for given arguments
def price_wrapper(dt, t, r, sigma, mu, s0, k, option_type):
    return stoch_price(dt, t, r, sigma, mu, s0, k, option_type)

# Greeks calculation function
def calculate_greeks(dt, t, r, sigma, mu, s0, k, option_type='call'):
    # Price at current values
    price = price_wrapper(dt, t, r, sigma, mu, s0, k, option_type)
    
    # Delta: Sensitivity to underlying asset price (s0)
    delta = finite_difference(price_wrapper, 5, 1e-5, dt, t, r, sigma, mu, s0, k, option_type)
    
    # Gamma: Sensitivity of delta (second derivative with respect to s0)
    gamma = finite_difference(price_wrapper, 5, 1e-5, dt, t, r, sigma, mu, s0, k, option_type)
    
    # Theta: Sensitivity to time (t)
    theta = -finite_difference(price_wrapper, 1, 1e-5, dt, t, r, sigma, mu, s0, k, option_type)
    
    # Vega: Sensitivity to volatility (sigma)
    vega = finite_difference(price_wrapper, 3, 1e-5, dt, t, r, sigma, mu, s0, k, option_type)
    
    # Rho: Sensitivity to interest rate (r)
    rho = finite_difference(price_wrapper, 2, 1e-5, dt, t, r, sigma, mu, s0, k, option_type)

    return {
        "Price": price,
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta,
        "Vega": vega,
        "Rho": rho
    }

# Example usage
# dt = 1/12  # Small time step
# t = 1      # Time to expiration (in years)
# r = 0.05   # Risk-free interest rate
# sigma = 0.2  # Volatility
# mu = 0.05  # Drift term
# s0 = 100   # Current stock price
# k = 100    # Strike price

# greeks = calculate_greeks(dt, t, r, sigma, mu, s0, k, option_type='call')
# print(greeks)