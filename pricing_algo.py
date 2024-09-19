import numpy as np
from bs import black_scholes#(<#S#>, <#K#>, <#T#>, <#r#>, <#sigma#>, <#option_type#>)

def derivative_price(prices, mu, alpha, sigma, leverage_factor=10):
	# Compute variance from the mean over time
	deviations = np.mean((prices - mu)**2)
	# The price of the derivative increases if the price adheres closely to the mean
	# i.e., the variance is minimized. We penalize high deviations and low reversion.
	# The larger alpha (reversion coefficient), the larger the derivative price,
	# the smaller sigma (spread coefficient), the larger the derivative price.
	derivative_prices = leverage_factor * alpha / (1 + sigma * np.sqrt(deviations))
	# Take the average price across simulations
	return np.mean(derivative_prices)

#mu = np.mean(df)  # mean reversion level
#alpha = 2.0  # reversion coefficient (higher means faster reversion)
#sigma = 0.5  # spread coefficient (lower means lower variance around the mean)
#r = 0.1  # population growth rate (logistic)
#K = 150  # carrying capacity for the population dynamics
#
#price = derivative_price(df, mu, alpha, sigma, .10)
#print(price)