#!/usr/bin/env python3

class TokenizedInterestRateSwap:
	def __init__(self, notional, fixed_rate, floating_rate_spread, counterparty_a, counterparty_b):
		self.notional = notional
		self.fixed_rate = fixed_rate
		self.floating_rate_spread = floating_rate_spread
		self.counterparty_a = counterparty_a
		self.counterparty_b = counterparty_b
		
	def calculate_hash(self, index, previous_hash, data):
		block_string = f"{index}{previous_hash}{data}"
		return hashlib.sha256(block_string.encode()).hexdigest()
	
	def create_genesis_block(self, swap_id):
		genesis_block = SwapBlock(
			index=0,
			timestamp=datetime.now(),
			data={'message': 'Genesis Block'},
			previous_hash='0',
			hash=self.calculate_hash(0, '0', 'Genesis Block'),
			swap_id=swap_id
		)
		db.session.add(genesis_block)
		db.session.commit()
		
	def add_transaction(self, swap_id, period, floating_rate_index, fixed_payment, floating_payment, net_cash_flow):
		transaction_data = {
			'period': period,
			'floating_rate_index': floating_rate_index,
			'fixed_payment': fixed_payment,
			'floating_payment': floating_payment,
			'net_cash_flow': net_cash_flow,
			'counterparty_a': self.counterparty_a,
			'counterparty_b': self.counterparty_b
		}
		previous_block = SwapBlock.query.filter_by(swap_id=swap_id).order_by(SwapBlock.index.desc()).first()
		if previous_block is None:
			self.create_genesis_block(swap_id)
			previous_block = SwapBlock.query.filter_by(swap_id=swap_id).order_by(SwapBlock.index.desc()).first()
		new_index = previous_block.index + 1
		new_block = SwapBlock(
			index=new_index,
			timestamp=datetime.now(),
			data=transaction_data,
			previous_hash=previous_block.hash,
			hash=self.calculate_hash(new_index, previous_block.hash, str(transaction_data)),
			swap_id=swap_id
		)
		db.session.add(new_block)
		db.session.commit()
		
	def simulate_payments(self, swap_id, floating_rate_indices):
		for period, floating_rate_index in enumerate(floating_rate_indices, start=1):
			fixed_payment = self.notional * self.fixed_rate
			floating_payment = self.notional * (floating_rate_index + self.floating_rate_spread)
			net_cash_flow = floating_payment - fixed_payment
			self.add_transaction(swap_id, period, floating_rate_index, fixed_payment, floating_payment, net_cash_flow)
			
			