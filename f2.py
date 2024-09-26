import yfinance as yf
import numpy as np
from scipy.integrate import quad
from scipy.stats import poisson


#ticker = input("ticker >>\t").upper()
#t = yf.Ticker(ticker)
#history = t.history(period='1y')
#close = history["Close"]
#ret = close.pct_change()[1:]# Percentage returns (ignoring first NaN)
#ret = [(i-np.mean(ret))/np.std(ret) for i in ret]

# Define the B(t) function
def B(t):
    return 1 / np.cosh(t)

# Define the W(t) function (Poisson-like distribution approximation)
def W(t):
    if t <= 0:
        return 0  # Handle invalid t values
    lambda_t = 1 / t
    poisson_dist = poisson(mu=lambda_t)
    return poisson_dist.pmf(int(t))  # t should be integer for Poisson PMF

# Define the f(t) function
def f(t):
    if t <= 0:
        return 0  # Handle invalid t values
    # Calculate the integrals
    integral_B = quad(B, 0, t)[0]
    integral_WB2 = quad(lambda y: W(y) * B(y)**2, 0, t)[0]
    
    # Define the full expression inside absolute value
    expr = (t**2 * B(t)**2 
            - 2 * t * B(t) * integral_B 
            + integral_WB2 * B(t)**2)
    
    # Return the absolute value
    return abs(expr)

## Apply f(t) on the returns
#r = np.array([f(i) for i in ret])
#r = np.sqrt(r@r)
#var = np.sqrt(sum([(i-np.mean(close))**2 for i in close]))
#print(r)
#print(var)
