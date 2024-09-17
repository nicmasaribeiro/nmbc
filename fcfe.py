#!/usr/bin/env python3

import numpy as np

def fcfe_dcf_model(free_cash_flows, cost_of_equity, terminal_growth_rate, forecast_years):
	"""
	Calculate the equity value using the FCFE DCF model.

	:param free_cash_flows: List of projected Free Cash Flow to Equity (FCFE) values for each year
	:param cost_of_equity: Required rate of return for equity (as a decimal)
	:param terminal_growth_rate: Growth rate after the forecast period (as a decimal)
	:param forecast_years: Number of years in the forecast period
	:return: Present value of equity (stock price)
	"""
	
	# Discount FCFEs to the present
	discounted_fcfes = [
		fcfe / ((1 + cost_of_equity) ** (t + 1))
		for t, fcfe in enumerate(free_cash_flows)
	]
	
	# Terminal Value calculation (using perpetuity growth model)
	terminal_value = (free_cash_flows[-1] * (1 + terminal_growth_rate)) / (cost_of_equity - terminal_growth_rate)
	
	# Discount the terminal value to the present
	discounted_terminal_value = terminal_value / ((1 + cost_of_equity) ** forecast_years)
	
	# Calculate the total equity value (sum of discounted FCFEs + discounted terminal value)
	equity_value = np.sum(discounted_fcfes) + discounted_terminal_value
	
	return equity_value
#
## Example usage:
## Projected Free Cash Flows to Equity for 5 years
#fcfes = [100, 110, 120, 130, 140]
#
## Cost of equity (e.g., 10%)
#cost_of_equity = 0.10
#
## Terminal growth rate (e.g., 3%)
#terminal_growth_rate = 0.03
#
## Number of forecast years
#forecast_years = 5
#
## Calculate the equity value (stock price)
#equity_value = fcfe_dcf_model(fcfes, cost_of_equity, terminal_growth_rate, forecast_years)
#print(f"Estimated Equity Value: ${equity_value:.2f}")
