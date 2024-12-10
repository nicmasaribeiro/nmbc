#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

def calculate_stock_price_and_derivatives(V, delta, K, T, r, sigma):
	"""
	Calculate the stock price and derivatives (time and maturity) based on the option value and Delta.
	
	Parameters:
		V (float): Option value.
		delta (float): Delta of the option (dV/dS).
		K (float): Strike price of the option.
		T (float): Time to maturity (in years).
		r (float): Risk-free interest rate.
		sigma (float): Volatility of the stock.
	
	Returns:
		tuple: Estimated stock price, time derivative (Theta), and maturity sensitivity.
	"""
	# Numerical derivative function
	def numerical_derivative(func, variable, dx=1e-5):
		return (func(variable + dx) - func(variable - dx)) / (2 * dx)
	
	# Delta equation to solve for stock price
	def delta_equation(S):
		d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
		calculated_delta = norm.cdf(d1)
		return calculated_delta - delta
	
	# Provide an initial guess for the stock price
	S_initial_guess = K
	
	# Solve for stock price using fsolve
	stock_price = fsolve(delta_equation, S_initial_guess)[0]
	
	# Define the option value function
	def option_value(S):
		d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
		d2 = d1 - sigma * np.sqrt(T)
		return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
	
	# Time derivative (Theta)
	def option_value_with_time(T_val):
		d1 = (np.log(stock_price / K) + (r + (sigma ** 2) / 2) * T_val) / (sigma * np.sqrt(T_val))
		d2 = d1 - sigma * np.sqrt(T_val)
		return stock_price * norm.cdf(d1) - K * np.exp(-r * T_val) * norm.cdf(d2)
	
	theta = numerical_derivative(option_value_with_time, T)
	
	# Maturity sensitivity
	def option_value_with_maturity(T_val):
		return option_value(stock_price)
	
	maturity_sensitivity = numerical_derivative(option_value_with_maturity, T)
	
	return stock_price, theta, maturity_sensitivity

# # Example Usage
# V = 11.49  # Option value
# delta = 0.31493887972544593  # Delta
# K = 100  # Strike price
# T = 1  # Time to maturity (1 year)
# r = 0.05  # Risk-free interest rate (5%)
# sigma = 0.2  # Volatility (20%)

# S, theta, maturity_sensitivity = calculate_stock_price_and_derivatives(V, delta, K, T, r, sigma)
# print(f"Calculated Stock Price: {S}")
# print(f"Time Derivative (Theta): {theta}")
# print(f"Maturity Sensitivity: {maturity_sensitivity}")
