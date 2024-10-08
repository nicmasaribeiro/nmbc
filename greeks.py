import numpy as np
import scipy.stats as si

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculate the Greeks for a European option using the Black-Scholes model.
    
    Parameters:
    S : float : Spot price of the underlying asset
    K : float : Strike price of the option
    T : float : Time to expiration in years
    r : float : Risk-free interest rate
    sigma : float : Volatility of the underlying asset
    option_type : str : Type of the option ("call" or "put")
    
    Returns:
    dict : Dictionary containing the option price and Greeks: Delta, Gamma, Theta, Vega, and Rho
    """

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate option price and Greeks for a call option
    if option_type == "call":
        option_price = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
        delta = si.norm.cdf(d1, 0.0, 1.0)
        rho = K * T * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif option_type == "put":
        option_price = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)
        delta = -si.norm.cdf(-d1, 0.0, 1.0)
        rho = -K * T * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    # Common Greeks
    gamma = si.norm.pdf(d1, 0.0, 1.0) / (S * sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1, 0.0, 1.0) * np.sqrt(T) / 100
    theta_call = -(S * si.norm.pdf(d1, 0.0, 1.0) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    theta_put = -(S * si.norm.pdf(d1, 0.0, 1.0) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)

    theta = theta_call if option_type == "call" else theta_put
    theta = theta / 365  # Per day decay

    return {
        "Option Price": option_price,
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta,
        "Vega": vega,
        "Rho": rho
    }

# # Example usage
# S = 100  # Current stock price
# K = 100  # Strike price
# T = 1    # Time to expiration in years
# r = 0.05 # Risk-free rate
# sigma = 0.2  # Volatility

# greeks = black_scholes_greeks(S, K, T, r, sigma, option_type="call")
# print(greeks)