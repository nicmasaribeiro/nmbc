from scipy.stats import norm
import numpy as np
from scipy.integrate import quad


# Define the modified Brownian motion process B(s) as the PDF of a normal distribution with mean 0 and variance 1/t
def B(s):
	return norm.pdf(s, loc=0, scale=1/np.sqrt(s))

# Define W(s) as a standard Brownian motion or Wiener process
def W(s, seed=42):
	np.random.seed(seed)
	return np.random.normal(0,1/s)
# Define the function f(s) again with the modified B(s)

def f(s):
	# First term: s^2 * B(s)^2
	first_term = s**2 * B(s)**2
	
	# Second term: -2s * B(s) * integral of B(t) from 0 to s
	def B_t_integral(t):
		return B(t)
	
	second_term, _ = quad(B_t_integral, 0, s)
	second_term = -2 * s * B(s) * second_term
	
	# Third term: integral from 0 to s of the integral of B(x)^2 from 0 to s
	def B_x_squared_integral(x):
		return B(x)**2
	
	inner_integral, _ = quad(B_x_squared_integral, 0, s)
	third_term, _ = quad(lambda y: inner_integral, 0, s)
	
	# Combine the terms
	return first_term + second_term + third_term

# Define the function g(t, r)
def g(t, r):
	return (1 - abs(r)) * np.exp(r * f(t))

# Define F(t) as per the provided expression
def F(t):
	# Define the integrand f(s)^2 * W(s)
	def integrand(s):
		return f(s)**2 * W(s)
	
	# Perform the integral from 0 to t
	integral_result, _ = quad(integrand, 0, t)
	
	# Calculate F(t)
	return integral_result**(1/t)

def S(t,r,dt):
	return (g(t,r)**t)*dt+np.exp(f(t))


def stoch_price(dt,t,r,sigma,mu,s0,k,option_type='call'):
	if option_type =="call":
		first =  F(t)*(S(t,r,dt)*s0-k)
		second = np.exp(r)*F(t)*(1-F(t)/k)
		third = np.exp((mu-(sigma**2)/2)*t+sigma*W(t))
		result = first+second+third
		return max(0,result)
	elif option_type == 'put':
		first =  F(t)*(S(t,r,dt)*k-s0)
		second = np.exp(r)*F(t)*(1-F(t)/k)
		third = np.exp((mu-(sigma**2)/2)*t+sigma*W(t))
		result = first+second+third
		return max(0,result)
#
#price = stoch_price(1/12,13, .1, .3, .05, 100, 80,'call')
#print(price)