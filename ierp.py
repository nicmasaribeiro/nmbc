# Function to calculate the implied equity risk premium
def calculate_implied_equity_risk_premium(expected_dividend, market_price, growth_rate, risk_free_rate):
    """
    This function calculates the implied equity risk premium using the Gordon Growth Model.
    
    :param expected_dividend: Expected dividend for next year (in currency units)
    :param market_price: Current market price (e.g., price of the stock index)
    :param growth_rate: Long-term growth rate of dividends (as a percentage, e.g., 5 for 5%)
    :param risk_free_rate: Risk-free rate (as a percentage, e.g., 2 for 2%)
    
    :return: Implied Equity Risk Premium (IERP)
    """
    expected_return = (expected_dividend / market_price) + growth_rate / 100
    implied_erp = expected_return - risk_free_rate / 100
    return implied_erp

# Example inputs
expected_dividend = 50.0  # Expected dividend for next year
market_price = 1000.0     # Current market price (e.g., stock index level)
growth_rate = 5.0         # Long-term growth rate (in %)
risk_free_rate = 2.0      # Risk-free rate (in %)

# Calculate Implied Equity Risk Premium
implied_erp = calculate_implied_equity_risk_premium(expected_dividend, market_price, growth_rate, risk_free_rate)
print(f"Implied Equity Risk Premium: {implied_erp * 100:.2f}%")
