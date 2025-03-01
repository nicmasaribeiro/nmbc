
import numpy as np

class TokenizedRateSwap:

	def __init__(self,issuer,swap_id, notional, fixed_rate, maturity, token_supply):
		"""
		Initialize a tokenized interest rate swap.

		Parameters:
		- swap_id: Identifier for the swap contract.
		- notional: Total notional amount of the swap.
		- fixed_rate: Fixed rate (annualized) paid on the swap.
		- maturity: Maturity date or term of the swap (could be a date string or number of periods).
		- token_supply: Total number of tokens representing fractional ownership of the swap.
		"""
		self.swap_id = swap_id
		self.notional = notional
		self.fixed_rate = fixed_rate
		self.maturity = maturity
		self.token_supply = token_supply
			
		# Blockchain-style token details:
		self.name = f"TokenizedSwap-{swap_id}"
		self.symbol = f"TS{swap_id}"
			
		# Ledger: dictionary mapping account identifiers to token balances.
		# Initially, all tokens are held by the issuer.
		self.issuer = issuer
		self.balances = {self.issuer: token_supply}
			
		print(f"Swap {swap_id} tokenized: {token_supply} tokens representing a notional of {notional}")
		
	def transfer(self, sender, receiver, amount):
		"""
		Simulate a token transfer between accounts.

		Parameters:
		- sender: The account sending tokens.
		- receiver: The account receiving tokens.
		- amount: Number of tokens to transfer.
		"""
		if self.balances.get(sender, 0) < amount:
			raise ValueError("Insufficient balance to transfer")
		self.balances[sender] -= amount
		self.balances[receiver] = self.balances.get(receiver, 0) + amount
		print(f"Transferred {amount} tokens from {sender} to {receiver}")
		
	def balance_of(self, account):
		"""
		Return the token balance of the given account.
		"""
		return self.balances.get(account, 0)
	
	def simulate_swap_cashflows(self, periods):
		"""
		Simulate simple net cash flows of the swap over a number of periods.
		
		For each period, the fixed leg pays a constant amount while the floating leg
		is simulated as a small random variation around the fixed payment.

		Parameters:
		- periods: Number of payment periods (e.g., years).
		
		Returns:
		- A list of net cash flows (floating - fixed) for each period.
		"""
		cash_flows = []
		for t in range(1, periods + 1):
			fixed_payment = self.notional * self.fixed_rate
			cash_flows.append(fixed_payment)
	
		floating_payment = fixed_payment * (1 + np.random.normal(0, 0.01))
		net_cash_flow = floating_payment - fixed_payment
		cash_flows.append(net_cash_flow)
		return cash_flows
      
#		TokenizedRateSwap(issuer, swap_id, notional, fixed_rate, maturity, token_supply)
token  = TokenizedRateSwap('nmr','123', 4_000_000, .05, 10, 100_000)
token.transfer('nmr', 'gzago', 10)
print(np.random.normal(0, 0.01))
print(token.simulate_swap_cashflows(4))