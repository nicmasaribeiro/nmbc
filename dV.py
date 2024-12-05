#!/usr/bin/env python3
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative
import warnings
import yfinance as yf
warnings.filterwarnings('ignore')


# t = 0 #np.linspace(0,3,10)
# s = 2 #np.linspace(3,5,10)
# T = 5 #np.linspace(5,10,10)
# r0 = .06
# c0 = 1/4
# K = 10


# Define B0 function
def B0(t, s, T):
	return lambda r0, c0: (r0 * np.exp(((T - t) * r0) / c0)) / ((s - t) * (T - t))
# Test B0

# Define V function
def V(t, s, T, K, r0, c0):
	"""
	Function to compute V.
	"""
	B0_value = B0(t, s, T)(r0, c0)
	term1 = B0_value * (1 - (B0_value / K))
	term2 = B0_value * K * np.exp(-r0 * t)
	return term1 + term2
# Test V

# Define dvdt function
def dVdt(t, s, T, K, r0, c0):
	"""
	Function to compute the derivative of V with respect to t numerically.
	"""
	# Create a lambda function for V(t, s, T, K, r0, c0) with respect to t
	V_func = lambda t: V(t, s, T, K, r0, c0)
	return derivative(V_func, t, dx=1e-5)
# Test dvdt

# Define dvdt function
def dVdt2(t, s, T, K, r0, c0):
	"""
	Function to compute the derivative of V with respect to t numerically.
	"""
	# Create a lambda function for V(t, s, T, K, r0, c0) with respect to t
	V_func = lambda t: V(t, s, T, K, r0, c0)
	return derivative(V_func, t, dx=1e-5,n=2)
# Test dvdt



def dVds(t, s, T, K, r0, c0):
	"""
	Function to compute the derivative of V with respect to s numerically.
	"""
	# Create a lambda function for V(t, s, T, K, r0, c0) with respect to s
	V_func = lambda s: V(t, s, T, K, r0, c0)
	return derivative(V_func, s, dx=1e-5)


def dVds2(t, s, T, K, r0, c0):
	"""
	Function to compute the derivative of V with respect to s numerically.
	"""
	# Create a lambda function for V(t, s, T, K, r0, c0) with respect to s
	V_func = lambda s: V(t, s, T, K, r0, c0)
	return derivative(V_func, s, dx=1e-5,n=2)


def dVdT(t, s, T, K, r0, c0):
	"""
	Function to compute the derivative of V with respect to T numerically.
	"""
	# Create a lambda function for V(t, s, T, K, r0, c0) with respect to T
	V_func = lambda T: V(t, s, T, K, r0, c0)
	return derivative(V_func, T, dx=1e-5)


def dVdT2(t, s, T, K, r0, c0):
	"""
	Function to compute the derivative of V with respect to T numerically.
	"""
	# Create a lambda function for V(t, s, T, K, r0, c0) with respect to T
	V_func = lambda T: V(t, s, T, K, r0, c0)
	return derivative(V_func, T, dx=1e-5,n=2)
# r_dvdT2 = dVdT2(t, s, T, K, r0, c0)

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
		result = (first+second+third)
		return max(0,result)
#
# Rt = ((1+r_dVdt)**(r_dVdt2))*((1+r_dVds)**(r_dVds2))*((1+r_dvdT)**(r_dvdT2))
