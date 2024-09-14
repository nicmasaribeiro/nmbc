#!/usr/bin/env python3

#!/usr/bin/env python3
import numpy as np
import pandas as pd
class DCF:
	def __init__(self,ni,depre,ebit,growth,wacc,taxes):
			self.ni = ni
			self.ebit = ebit
			self.depre = depre
			self.taxes = taxes
			self.growth = growth
			self.wacc = wacc
			self.growth_terminal = 0.025
		
	def rev(self):
			ls = []
			for i in range(0, 5):
					r = self.ebit * (1 + self.growth) ** i
					ls.append(r)
			return np.array(ls)
	
	#	result = np.array(rev(revenue,0.24))
	
	def sheet(self, result):
			A = np.array([result, result,result,result])
			t = np.array([
				[1,self.ni,self.depre,self.taxes], [1,self.ni,self.depre,self.taxes],
						[1,self.ni,self.depre,self.taxes], [1,self.ni,self.depre,self.taxes],
								[1,self.ni,self.depre,self.taxes]])
			return t.T * A
	
	
	def calculate_cashflow(self, result):
			fcffs = []
			for i in range(0, 5):
					fcff = self.sheet(result).T[i][0] - self.sheet(result).T[i][1] + self.sheet(result).T[i][2] - self.sheet(result).T[i][3]
					fcffs.append(fcff)
			return fcffs
	
	
	def final(self, cf):
			discounted = [cf[i] / (1 + self.wacc) ** i for i in range(0, 5)]
			tv = (discounted[0]*(1+self.growth_terminal))/ (self.wacc - self.growth_terminal)
			free_cash_flow = np.sum(discounted)+tv
			return free_cash_flow
	
	
