#!/usr/bin/env python3

#!/usr/bin/env python3

import requests
import json
import pandas as pd
import numpy as np
class ProfiMainEquity():
	
	def get_netIncome(self,ticker):
		url = "https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=120&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
		
		response = requests.request("GET", url)
		data = json.loads(response.text)
		# print(data[0]['totalCurrentLiabilities'])
		df =[]
		for i in range(0,5):
			df.append(data[i]['netIncome'])
		return df
	#
	# print(get_workingCapital('AAPL'))
	
	def get_SHE(self,ticker):
		url = "https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?limit=120&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
		
		response = requests.request("GET", url)
		data = json.loads(response.text)
		# print(data[0]['totalCurrentLiabilities'])
		df =[]
		for i in range(0,5):
			df.append(data[i]['totalStockholdersEquity'])
			
		return df
	# print(get_operatingIncome('AAPL'))
	
	def roe(self,ticker):
		roe = []
		for i in range(0,4):
			r = self.get_SHE(ticker)[i+1]/self.get_netIncome(ticker)[i]
			roe.append(r)
		return (roe)
	
	def get_capex(self,ticker):
		url = "https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?limit=120&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
		
		response = requests.request("GET", url)
		data = json.loads(response.text)
		# print(data[0]['totalCurrentLiabilities'])
		df =[]
		for i in range(0,5):
			df.append(data[i]['capitalExpenditure'])
			
		return df
	
	def capexMargin(self,ticker):
		margin = []
		for i in range(0,4):
			r = self.get_capex(ticker)[i+1]/self.get_netIncome(ticker)[i]
			margin.append(r)
	
		return np.mean(margin)
	
		
#s = ProfiMain()
#print(s.roc('MSFT'))
	