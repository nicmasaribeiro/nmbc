import numpy as np
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
	"""
	Calculate the Black-Scholes option price.

	Parameters:
	- S: Current stock price
	- K: Strike price
	- T: Time to expiration (in years)
	- r: Risk-free interest rate
	- sigma: Volatility of the stock (annualized)
	- option_type: "call" or "put"

	Returns:
	- Option price
	"""
	d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
	d2 = d1 - sigma * np.sqrt(T)
	if option_type == "call":
		return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
	elif option_type == "put":
		return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
	else:
		raise ValueError("Invalid option type. Use 'call' or 'put'.")
		
def binary_search_stock_price(option_price, K, T, r, sigma, option_type="call", tol=1e-6):
	"""
	Estimate the current stock price using binary search.

	Parameters:
	- option_price: Observed market price of the option
	- K: Strike price
	- T: Time to expiration (in years)
	- r: Risk-free interest rate (annualized)
	- sigma: Volatility of the stock (annualized)
	- option_type: "call" or "put"
	- tol: Tolerance for convergence

	Returns:
	- Estimated stock price (S)
	"""
	# Define the bounds for the binary search
	lower_bound = 0.01  # Minimum possible stock price
	upper_bound = 2 * K  # A reasonable maximum for the stock price
	
	while upper_bound - lower_bound > tol:
		# Calculate the midpoint
		mid = (lower_bound + upper_bound) / 2
		
		# Calculate the option price for the midpoint
		calculated_price = black_scholes_price(mid, K, T, r, sigma, option_type)
		
		# Narrow the range based on the comparison
		if calculated_price < option_price:
			lower_bound = mid  # Stock price is higher
		else:
			upper_bound = mid  # Stock price is lower
			
	# Return the midpoint as the estimated stock price
	return (lower_bound + upper_bound) / 2

# # Example parameters
# option_price = 3  # Observed call option price
# K = 105             # Strike price
# T = 1               # Time to expiration (1 year)
# r = 0.05            # Risk-free interest rate (5%)
# sigma = 0.2         # Volatility (20%)

# # Solve for stock price
# stock_price = binary_search_stock_price(option_price, K, T, r, sigma, option_type="call")
# print(f"Estimated Stock Price: {stock_price:.2f}")