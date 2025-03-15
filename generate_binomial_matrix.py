#!/usr/bin/env python3

import numpy as np

def generate_binomial_matrix(S0, u, d, r, T, n):
	"""
	Generate a binomial matrix of expected stock values.

	Parameters:
		S0 (float): Current stock price
		u (float): Upward movement factor
		d (float): Downward movement factor
		r (float): Risk-free rate (annual)
		T (float): Time to maturity (in years)
		n (int): Number of time steps

	Returns:
		numpy.ndarray: Binomial matrix of stock prices
	"""
	# Time step size
	delta_t = T / n
	
	# Risk-neutral probability
	p = (np.exp(r * delta_t) - d) / (u - d)
	
	# Initialize the binomial matrix
	binomial_matrix = np.zeros((n + 1, n + 1))
	
	# Fill in the stock prices at each node
	for i in range(n + 1):
		for j in range(i + 1):
			binomial_matrix[j, i] = S0 * (u ** (i - j)) * (d ** j)
			
	return binomial_matrix
#
## Example usage:
#S0 = 100   # Initial stock price
#u = 1.1    # Up factor
#d = 0.9    # Down factor
#r = 0.05   # Risk-free rate
#T = 1      # Time to maturity in years
#n = 5      # Number of time steps
#
#binomial_matrix = generate_binomial_matrix(S0, u, d, r, T, n)
#print("Binomial Matrix of Stock Prices:")
#print(binomial_matrix)
