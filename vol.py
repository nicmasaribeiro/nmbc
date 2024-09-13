import scipy.stats as si
import numpy as np

# Black-Scholes option pricing model for Call and Put
def black_scholes_option(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes option price.
    
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility (standard deviation of stock returns)
    option_type: 'call' for Call option, 'put' for Put option
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = (S * si.norm.cdf(d1, 0.0, 1.0) - 
                 K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    elif option_type == 'put':
        price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) -
                 S * si.norm.cdf(-d1, 0.0, 1.0))
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    
    return price

# Implied volatility calculation using Newton-Raphson method
def implied_volatility_option(S, K, T, r, market_price, option_type='call', tol=1e-5, max_iter=100):
    """
    Calculate the implied volatility using the market price of an option.
    
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    market_price: Observed market price of the option
    option_type: 'call' for Call option, 'put' for Put option
    tol: Tolerance for the convergence of volatility
    max_iter: Maximum number of iterations for Newton-Raphson method
    """
    # Initial guess for volatility
    sigma = 0.2
    for i in range(max_iter):
        # Calculate option price using current sigma
        price = black_scholes_option(S, K, T, r, sigma, option_type)
        # Calculate vega (derivative of the price with respect to sigma)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * si.norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
        
        # Newton-Raphson step
        price_diff = price - market_price
        if abs(price_diff) < tol:
            return sigma
        sigma -= price_diff / vega
    
    # If not converged, return the last estimate
    return sigma

# Example usage
# S = 100  # Stock price
# K = 100  # Strike price
# T = 1    # Time to maturity (in years)
# r = 0.05 # Risk-free interest rate
# market_price_call = 10  # Market price of the call option
# market_price_put = 7    # Market price of the put option

# # Calculate implied volatility for Call and Put options
# implied_vol_call = implied_volatility_option(S, K, T, r, market_price_call, option_type='call')
# implied_vol_put = implied_volatility_option(S, K, T, r, market_price_put, option_type='put')

# print(f"The implied volatility for the call option is {implied_vol_call:.4f}")
# print(f"The implied volatility for the put option is {implied_vol_put:.4f}")
