import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
from scipy.misc import derivative
import warnings
warnings.filterwarnings('ignore')

# # Example Usage
# V = 10  # Option value
# delta = 0.6  # Delta
# K = 100  # Strike price
# T = 1.5  # Time to maturity (1 year)
# s = 1
# t = 0
# r = 0.05  # Risk-free interest rate (5%)
# v = 0.2  # Volatility (20%)
# St = 100


# def d1(St,K,r,v,t,s,T):
# 	return (np.log(St/K)+((r+v**2/2)*(T-t)/(s-t)*(T-t)))/v*np.sqrt((T-t)*(T-s))
def d1(St,K,r,v,t,s,T):
	return (np.log(St/K)+((r+v**2/2)*(T-t)**((s-t)*(T-s))))/v*np.sqrt((T-t)*(T-s))
# print(d1(St,K,r,v,t,s,T))

def d2(St,K,r,v,t,s,T):
	return d1(St, K, r, v, t, s, T) - v*np.sqrt((T-t)*(T-s))
# print(d2(St, K, r, v, t, s, T))

def price(St,K,r,v,t,s,T):
	term1 = norm.cdf(d1(St, K, r, v, t, s, T))*St
	term2 = norm.cdf(d2(St, K, r, v, t, s, T))*K*np.exp((-r*(T-t))/((T-s)**(t-s)))
	return term1 - term2
# print("Price\t",price(St, K, r, v, t, s, T))

def dPdt(St,K,r,v,t,s,T):
	V_func = lambda t: price(St, K, r, v, t, s, T)
	return derivative(V_func, t, dx=1e-5)
# print("dPdt\t", dPdt(St,K,r,v,t,s,T))

def dPdt2(St,K,r,v,t,s,T):
	V_func = lambda t: price(St, K, r, v, t, s, T)
	return derivative(V_func, t, dx=1e-5,n=2)
# print("dPdt2\t", dPdt2(St,K,r,v,t,s,T))

def dPds(St,K,r,v,t,s,T):
	V_func = lambda s: price(St, K, r, v, t, s, T)
	return derivative(V_func, s, dx=1e-5)
# print("dPds\t", dPds(St,K,r,v,t,s,T))

def dPds2(St,K,r,v,t,s,T):
	V_func = lambda s: price(St, K, r, v, t, s, T)
	return derivative(V_func, s, dx=1e-5,n=2)
# print("dPds2\t", dPds2(St,K,r,v,t,s,T))

def dPdT(St,K,r,v,t,s,T):
	V_func = lambda T: price(St, K, r, v, t, s, T)
	return derivative(V_func,T, dx=1e-5)
# print("dPdT\t", dPdT(St,K,r,v,t,s,T))

def dPdT2(St,K,r,v,t,s,T):
	V_func = lambda T: price(St, K, r, v, t, s, T)
	return derivative(V_func,T, dx=1e-5,n=2)
# print("dPdT2\t", dPdT2(St,K,r,v,t,s,T))

def dPdr(St,K,r,v,t,s,T):
	V_func = lambda r: price(St, K, r, v, t, s, T)
	return derivative(V_func, r, dx=1e-5)
# print("dPdr\t", dPdr(St,K,r,v,t,s,T))

def dPdv(St,K,r,v,t,s,T):
	V_func = lambda v: price(St, K, r, v, t, s, T)
	return derivative(V_func, v, dx=1e-5)
# print("dPdv\t", dPdv(St,K,r,v,t,s,T))

