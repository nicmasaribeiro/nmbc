from scipy.optimize import fsolve
from scipy.stats import norm
import numpy as np
from scipy.misc import derivative
import warnings
warnings.filterwarnings('ignore')

def black_scholes_call_price(s, K, T, t, ts, r, sigma):
	"""
	Calculate the Black-Scholes price for a European call option.

	Parameters:
	- s: Stock price
	- K: Strike price
	- T: Maturity time
	- t: Current time
	- r: Risk-free interest rate
	- sigma: Volatility (standard deviation)

	Returns:
	- Call option price
	"""
	tau = T - t
	d1 = (np.log(s / K) + (r + 0.5 * sigma ** 2) * tau**((T - t)*(t-ts))) / (sigma * np.sqrt(tau*(t - ts)))
	d2 = d1 - sigma * np.sqrt(tau*(T-ts))
	call_price = s * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
	return call_price

# Function to estimate stock price using Black-Scholes model
def estimate_stock_price_bs(option_price, K, T, t,ts, r, sigma):
	"""
	Estimate the stock price using the Black-Scholes model.

	Parameters:
	- option_price: Observed market option price
	- K: Strike price
	- T: Maturity time
	- t: Current time
	- r: Risk-free interest rate
	- sigma: Volatility (standard deviation)

	Returns:
	- Estimated stock price
	"""
	# Define the function to find the root
	def bs_difference(s):
		return black_scholes_call_price(s, K, T, t,ts, r, sigma) - option_price
	
	# Initial guess for stock price
	s_initial_guess = K
	
	# Solve for s
	s_estimated, info, ier, mesg = fsolve(bs_difference, s_initial_guess, full_output=True)
	
	if ier != 1:
		print("Root-finding did not converge:", mesg)
		return None
	
	return s_estimated[0]

# Define dvdt function
def dVdt(option_price, K, T, t, ts, r0, sigma):
	"""
	Function to compute the derivative of V with respect to t numerically.
	"""
	# Create a lambda function for V(t, s, T, K, r0, c0) with respect to t
	V_func = lambda t: estimate_stock_price_bs(option_price, K, T, t, ts, r0, sigma)
	return derivative(V_func, t, dx=1e-5)

# Test dvdt
# Define dvdt function
def dVdts(option_price, K, T, t, ts, r0, sigma):
	"""
	Function to compute the derivative of V with respect to t numerically.
	"""
	# Create a lambda function for V(t, s, T, K, r0, c0) with respect to t
	V_func = lambda ts: estimate_stock_price_bs(option_price, K, T, t, ts, r0, sigma)
	return derivative(V_func, ts, dx=1e-5)

# Test dvdt
# Define dvdt function
def dVdT(option_price, K, T, t, ts, r0, sigma):
	"""
	Function to compute the derivative of V with respect to t numerically.
	"""
	# Create a lambda function for V(t, s, T, K, r0, c0) with respect to t
	V_func = lambda T: estimate_stock_price_bs(option_price, K, T, t, ts, r0, sigma)
	return derivative(V_func, T, dx=1e-5)

# Test dvdt
# Define dvdt function
def dVdv(option_price, K, T, t, ts, r0, sigma):
	"""
	Function to compute the derivative of V with respect to t numerically.
	"""
	# Create a lambda function for V(t, s, T, K, r0, c0) with respect to t
	V_func = lambda sigma: estimate_stock_price_bs(option_price, K, T, t, ts, r0, sigma)
	return derivative(V_func, sigma, dx=1e-5)
# Test dvdt

# Define ∂V/∂K function
def dVdK(option_price, K, T, t, ts, r0, sigma):
	"""
	Function to compute the derivative of V with respect to t numerically.
	"""
	# Create a lambda function for V(t, s, T, K, r0, c0) with respect to t
	V_func = lambda K: estimate_stock_price_bs(option_price, K, T, t, ts, r0, sigma)
	return derivative(V_func, K, dx=1e-5)
# Test dvdt
# Define dvdt function
def dVdP(option_price, K, T, t, ts, r0, sigma):
	"""
	Function to compute the derivative of V with respect to t numerically.
	"""
	# Create a lambda function for V(t, s, T, K, r0, c0) with respect to t
	V_func = lambda option_price: estimate_stock_price_bs(option_price, K, T, t, ts, r0, sigma)
	return derivative(V_func, option_price, dx=1e-5)
# Test dvdt

# Define dvdP2 function
def dVdP2(option_price, K, T, t, ts, r0, sigma):
	"""
	Function to compute the derivative of V with respect to t numerically.
	"""
	# Create a lambda function for V(t, s, T, K, r0, c0) with respect to t
	V_func = lambda option_price: estimate_stock_price_bs(option_price, K, T, t, ts, r0, sigma)
	return derivative(V_func, option_price, dx=1e-5,n=2)
# Test dvdt

# Define dvdt function
def dVdr0(option_price, K, T, t, ts, r0, sigma):
	"""
	Function to compute the derivative of V with respect to t numerically.
	"""
	# Create a lambda function for V(t, s, T, K, r0, c0) with respect to t
	V_func = lambda r0: estimate_stock_price_bs(option_price, K, T, t, ts, r0, sigma)
	return derivative(V_func, r0, dx=1e-5)
# Test dvdt

def option_value(option_price, K, T, t, ts, r0, sigma):
	result = dVdts(option_price, K, T, t, ts, r0, sigma)+(1/2)*sigma*dVdP2(option_price, K, T, t, ts, r0, sigma)+r0*option_price*dVdP(option_price, K, T, t, ts, r0, sigma) - r0*estimate_stock_price_bs(option_price, K, T, t, ts, r0, sigma)
	return result

def option_value_dT(option_price, K, T, t, ts, r0, sigma):
	result = dVdT(option_price, K, T, t, ts, r0, sigma)+(1/2)*sigma*dVdP2(option_price, K, T, t, ts, r0, sigma)+r0*option_price*dVdP(option_price, K, T, t, ts, r0, sigma) - r0*estimate_stock_price_bs(option_price, K, T, t, ts, r0, sigma)
	return result

def option_value_dt(option_price, K, T, t, ts, r0, sigma):
	result = dVdt(option_price, K, T, t, ts, r0, sigma)+(1/2)*sigma*dVdP2(option_price, K, T, t, ts, r0, sigma)+r0*option_price*dVdP(option_price, K, T, t, ts, r0, sigma) - r0*estimate_stock_price_bs(option_price, K, T, t, ts, r0, sigma)
	return result



