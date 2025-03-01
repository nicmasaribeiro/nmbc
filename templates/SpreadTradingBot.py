#!/usr/bin/env python3

import random
import pandas as pd

class SpreadTradingBot:
	def __init__(self, spread_threshold=0.02, take_profit=0.05, stop_loss=0.03):
		"""
		Initializes the trading bot with given thresholds.
		:param spread_threshold: Minimum spread required to trigger a buy.
		:param take_profit: Profit percentage at which to sell.
		:param stop_loss: Loss percentage at which to sell.
		"""
		self.spread_threshold = spread_threshold
		self.take_profit = take_profit
		self.stop_loss = stop_loss
		self.position = None  # Holds the price at which we bought
		self.trade_log = []
		
	def get_market_data(self):
		"""Simulate fetching market data with bid-ask prices"""
		bid_price = round(random.uniform(99, 101), 2)
		ask_price = round(random.uniform(101, 103), 2)
		return bid_price, ask_price
	
	def check_trade_conditions(self, bid_price, ask_price):
		"""
		Determines whether to buy or sell based on spread and profit/loss conditions.
		"""
		spread = ask_price - bid_price
		
		# Buy condition: Spread is above threshold
		if spread >= self.spread_threshold and self.position is None:
			self.position = ask_price  # Buying at the ask price
			self.trade_log.append(f"BUY at {ask_price}")
			
		# Sell condition: If a position is open, sell based on take-profit or stop-loss
		elif self.position is not None:
			profit_loss = (bid_price - self.position) / self.position  # Profit or loss percentage
			
			if profit_loss >= self.take_profit or profit_loss <= -self.stop_loss:
				self.trade_log.append(f"SELL at {bid_price} | P&L: {profit_loss * 100:.2f}%")
				self.position = None  # Close position
				
	def run_simulation(self, steps=50):
		"""Runs a simulation of market conditions"""
		data = []
		for _ in range(steps):
			bid, ask = self.get_market_data()
			self.check_trade_conditions(bid, ask)
			data.append({"Bid": bid, "Ask": ask, "Spread": ask - bid})
			
		df = pd.DataFrame(data)
		print("\n".join(self.trade_log))
		return df
	
# Run simulation
bot = SpreadTradingBot()
df_results = bot.run_simulation(steps=50)

# Display results
import ace_tools_open as tools
tools.display_dataframe_to_user(name="Spread Trading Simulation", dataframe=df_results)
