from scipy.integrate import quad
import numpy as np
from scipy.stats import norm

# Define the risk-neutral density function (e.g., derived from Black-Scholes)
def risk_neutral_density(S, K, T, r, sigma):
	d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
	return norm.pdf(d1)

# Define the call option payoff function
def call_payoff(S, K):
	return max(S - K, 0)

# Define the integrand for option price calculation
def option_price_integrand(K, S0, T, r, sigma):
	density = risk_neutral_density(S0, K, T, r, sigma)
	payoff = call_payoff(S0, K)
	return payoff * density

# Compute the implied stock price using the current option price
def compute_stock_price_from_option(option_price, T, r, K_min, K_max, sigma):
	def integrand(K, S0):
		return option_price_integrand(K, S0, T, r, sigma)
	
	# Solve for S0 using numerical integration and optimization
	from scipy.optimize import minimize_scalar
	
	def objective(S0):
		integral, _ = quad(integrand, K_min, K_max, args=(S0,))
		calculated_option_price = np.exp(-r * T) * integral
		return abs(calculated_option_price - option_price)
	
	# Minimize the difference to find the implied stock price
	result = minimize_scalar(objective, bounds=(K_min, K_max), method='bounded')
	return result.x if result.success else None

# # Parameters
# option_price = .05  # Example option price
# T = 1.0833  # Time to maturity (1 year)
# r = 0.046  # Risk-free rate (5%)
# K_min = .05  # Minimum strike price
# K_max = 5  # Maximum strike price
# sigma = 0.1250  # Volatility (20%)

# # Calculate the implied stock price
# implied_stock_price = compute_stock_price_from_option(option_price, T, r, K_min, K_max, sigma)
# print(f"Implied Stock Price: {implied_stock_price}")
