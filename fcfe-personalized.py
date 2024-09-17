#!/usr/bin/env python3

import numpy as np


class FCFE:
	def __init__(self, capex,RND,interest,netIncome,growth,wacc):
			self.capex = capex
			self.RND = RND
			self.interest = interest
			self.netIncome = netIncome
			self.growth = growth
			self.wacc = wacc
			self.growth_terminal = 0.025
		
	def rev(self):
			ls = []
			for i in range(0, 5):
					r = self.netIncome * (1 + self.growth) ** i
					ls.append(r)
			return np.array(ls)
	
	
	def sheet(self, result):
			A = np.array([result, result,result,result])
			t = np.array([
						[1,self.capex,self.RND,self.interest], 
						[1,self.capex,self.RND,self.interest],
						[1,self.capex,self.RND,self.interest], 
						[1,self.capex,self.RND,self.interest],
						[1,self.capex,self.RND,self.interest]])
			return t.T * A
	
	
	def calculate_cashflow(self, result):
			fcfes = []
			for i in range(0, 4):
					fcfe = self.sheet(result).T[i][0] - self.sheet(result).T[i][1] - self.sheet(result).T[i][2]
					fcfes.append(fcfe)
			return fcfes

	def final(self, cf):
			discounted = [cf[i] / (1 + self.wacc) ** i for i in range(0, 4)]
			tv = (discounted[0]*(1+self.growth_terminal))/ (self.wacc - self.growth_terminal)
			free_cash_flow = np.sum(discounted)+tv
			return free_cash_flow
	