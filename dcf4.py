#!/usr/bin/env python3

#!/usr/bin/env python3
import numpy as np
import pandas as pd


class DCF:
	def __init__(self,oi,capex,interest,depre,growth,wacc,taxes,wc):
			self.oi = oi
			self.interest = interest
			self.capex = capex
			self.wc = wc
			self.depre = depre
			self.taxes = taxes
			self.growth = growth
			self.wacc = wacc
			self.growth_terminal = 0.025
		
	def rev(self):
			ls = []
			for i in range(0, 5):
					r = self.oi * (1 + self.growth) ** i
					ls.append(r)
			return np.array(ls)
	
	#	result = np.array(rev(revenue,0.24))
	
	def sheet(self, result):
			A = np.array([result, result,result,result,result,result])
			t = np.array([
				[1,self.capex,self.interest,self.depre,self.taxes,self.wc], 
				[1,self.capex,self.interest,self.depre,self.taxes,self.wc],
				[1,self.capex,self.interest,self.depre,self.taxes,self.wc], 
				[1,self.capex,self.interest,self.depre,self.taxes,self.wc],
				[1,self.capex,self.interest,self.depre,self.taxes,self.wc]])
			return t.T * A
	
	
	def calculate_cashflow(self, result):
			fcffs = []
			for i in range(0, 5):
					fcff = self.sheet(result).T[i][0] - self.sheet(result).T[i][1] - self.sheet(result).T[i][2] + self.sheet(result).T[i][3] -  self.sheet(result).T[i][4] - self.sheet(result).T[i][5]
					fcffs.append(fcff)
			return fcffs
	
	
	def final(self, cf):
			discounted = [cf[i] / (1 + self.wacc) ** i for i in range(0, 5)]
			tv = (discounted[0]*(1+self.growth_terminal))/ (self.wacc - self.growth_terminal)
			free_cash_flow = np.sum(discounted)+tv
			return free_cash_flow
		
d=DCF(0.5, 0.3, 0.1, 0.1, 300, 0.06, 0.05, 0.1)
#d.rev()