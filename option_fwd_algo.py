#!/usr/bin/env python3

#!/usr/bin/env python3

import math
from scipy.stats import norm
from scipy.misc import derivative
import warnings

warnings.filterwarnings('ignore')

def option_valuation_with_forward_rate(S0, K, r1, r2, T1, T2, tf, sigma, option_type):
	# Step 1: Calculate implied forward rate
	forward_rate = ((1 + r2)**T2 / (1 + r1)**T1)**(1 / (T2 - T1)) - 1
	
	# Step 2: Calculate forward price
	forward_price = S0 * (1 + forward_rate)**tf
	
	# Step 3: Compute d1 and d2 for Black-Scholes formula
	d1 = (math.log(forward_price / K) + (sigma**2 * tf) / 2) / (sigma * math.sqrt(tf))
	d2 = d1 - sigma * math.sqrt(tf)
	
	# Step 4: Option pricing
	if option_type.lower() == "call":
		option_price = math.exp(-r1 * T1) * (forward_price * norm.cdf(d1) - K * norm.cdf(d2))
	elif option_type.lower() == "put":
		option_price = math.exp(-r1 * T1) * (K * norm.cdf(-d2) - forward_price * norm.cdf(-d1))
	else:
		raise ValueError("Invalid option type. Use 'call' or 'put'.")
		
	return option_price

# # Example parameters
# S0 = 100  # Spot price
# K = 90   # Strike price
# r1 = 0.03 # Rate for shorter period
# r2 = 0.05 # Rate for longer period
# T1 = .0833    # Shorter period in years
# T2 = 1    # Longer period in years
# tf = 1.5  # Forward period in years
# sigma = 0.42  # Volatility
# option_type = "call"

# Calculate option value
# option_value = option_valuation_with_forward_rate(S0, K, r1, r2, T1, T2, tf, sigma, option_type)
# print(f"The value of the {option_type} option is: {option_value:.2f}")

def dPdT1(S0, K, r1, r2, T1, T2, tf, sigma, option_type):
	V_func = lambda T1: option_valuation_with_forward_rate(S0, K, r1, r2, T1, T2, tf, sigma, option_type)
	return derivative(V_func, T1, dx=1e-5)
# print("dPdT1\t", dPdT1(S0, K, r1, r2, T1, T2, tf, sigma, option_type))

def dPdT1_2(S0, K, r1, r2, T1, T2, tf, sigma, option_type):
	V_func = lambda T1: option_valuation_with_forward_rate(S0, K, r1, r2, T1, T2, tf, sigma, option_type)
	return derivative(V_func, T1, dx=1e-5,n=2)
# print("dPdT1_2\t", dPdT1_2(S0, K, r1, r2, T1, T2, tf, sigma, option_type))

def dPdT2(S0, K, r1, r2, T1, T2, tf, sigma, option_type):
	V_func = lambda T2: option_valuation_with_forward_rate(S0, K, r1, r2, T1, T2, tf, sigma, option_type)
	return derivative(V_func, T2, dx=1e-5)
# print("dPdT2\t", dPdT2(S0, K, r1, r2, T1, T2, tf, sigma, option_type))

def dPdT2_2(S0, K, r1, r2, T1, T2, tf, sigma, option_type):
	V_func = lambda T2: option_valuation_with_forward_rate(S0, K, r1, r2, T1, T2, tf, sigma, option_type)
	return derivative(V_func, T2, dx=1e-5,n=2)
# print("dPdT2_2\t", dPdT2_2(S0, K, r1, r2, T1, T2, tf, sigma, option_type))
