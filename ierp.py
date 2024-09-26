# Function to calculate the implied equity risk premium
def calculate_implied_equity_risk_premium(expected_dividend, market_price, growth_rate, risk_free_rate):
    expected_return = (expected_dividend / market_price) + growth_rate / 100
    implied_erp = expected_return - risk_free_rate / 100
    return implied_erp

## Example inputs
#expected_dividend = 50.0  # Expected dividend for next year
#market_price = 1000.0     # Current market price (e.g., stock index level)
#growth_rate = 5.0         # Long-term growth rate (in %)
#risk_free_rate = 2.0      # Risk-free rate (in %)
#
## Calculate Implied Equity Risk Premium
#implied_erp = calculate_implied_equity_risk_premium(expected_dividend, market_price, growth_rate, risk_free_rate)
#print(f"Implied Equity Risk Premium: {implied_erp * 100:.2f}%")
