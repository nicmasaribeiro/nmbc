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
    # Prevent divide-by-zero or other computational errors
    if T <= 0 or S <= 0 or sigma <= 0:
        raise ValueError("Time to expiration, spot price, and volatility must be positive.")
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Cumulative distribution and probability density functions
    N_d1 = si.norm.cdf(d1)
    N_d2 = si.norm.cdf(d2)
    N_neg_d1 = si.norm.cdf(-d1)
    N_neg_d2 = si.norm.cdf(-d2)
    pdf_d1 = si.norm.pdf(d1)
    
    # Option price and Greeks calculations
    if option_type == "call":
        option_price = S * N_d1 - K * np.exp(-r * T) * N_d2
        delta = N_d1
        rho = K * T * np.exp(-r * T) * N_d2
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2) / 365
    elif option_type == "put":
        option_price = K * np.exp(-r * T) * N_neg_d2 - S * N_neg_d1
        delta = -N_neg_d1
        rho = -K * T * np.exp(-r * T) * N_neg_d2
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * N_neg_d2) / 365
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Common Greeks
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100  # Vega is scaled to percentage change
    
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