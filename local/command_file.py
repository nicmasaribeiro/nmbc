#!/usr/bin/env python3
from scipy.misc import derivative
import numpy as np
from scipy.integrate import tplquad
import scipy.integrate as spi

b0 = 2.3
b1 = -.34
b2 =.4
a = .3

f = lambda t,s,a: ( b0 + b1*(( 1 - np.exp(-a * t)) /(a*t) )+ b2 * ((( 1 - np.exp(-a * t)) / (a*t) ) - np.exp(-a*t)))**s
print(f(1,2,4))


# Define the integration limits
def h1(x, y): return 0       # Lower limit for z
def h2(x, y): return 1   # Upper limit for z
	
def g1(x): return 0          # Lower limit for y
def g2(x): return 1      # Upper limit for y
	
#a, b = 0, 1   Limits for x

# Compute the integral
Ft = lambda t: spi.tplquad(f, 0, t, g1, g2, h1, h2)[0]
print("Triple Integral Result:", Ft(2))

dfdt = lambda t: (a * np.exp(-a*t) * t - 1 + np.exp(-a*t))/(a*t**2)
print("dfdt\t",dfdt(-1))

#def d1(t,s,a):
#	V_func = lambda t,s,a: f(t,s,a)
#	return derivative(V_func, t, dx=1e-5)
#print(d1(1,1,1))